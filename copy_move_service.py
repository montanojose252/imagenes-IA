#!/usr/bin/env python3
"""
copy_move_service.py — Microservicio de detección Copy-Move
IVS v3.0 — Image Verification System

Detecta si una región de la imagen fue copiada y pegada en otra zona
de la misma imagen (técnica común para tapar fechas, carteles, objetos).

Uso:
    pip install flask pillow numpy scipy
    python copy_move_service.py

Llamada desde PHP:
    POST http://localhost:5001/analyze
    Body: { "image_base64": "...", "filename": "foto.jpg" }

Respuesta:
    {
        "detected": true,
        "confidence": 87,
        "regions": [
            { "source": [x1,y1,w,h], "destination": [x2,y2,w,h], "similarity": 94 }
        ],
        "heatmap_base64": "...",
        "summary": "Se detectaron 2 región(es) copiadas..."
    }
"""

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
import os
import hashlib
from scipy.spatial.distance import cdist

app = Flask(__name__)

# ── Configuración ─────────────────────────────────────────
BLOCK_SIZE   = 16    # tamaño de bloque para comparación (px)
STEP         = 8     # paso entre bloques (overlap 50%)
SIM_THRESHOLD = 0.92 # similitud mínima para considerar copia (0-1)
MIN_DISTANCE  = 50   # distancia mínima entre bloques para evitar falsos positivos
MAX_IMAGE_DIM = 800  # redimensionar imágenes grandes para eficiencia
SECRET_KEY    = os.environ.get('IVS_SERVICE_KEY', 'ivs_copymove_2026')


def resize_if_needed(img, max_dim=MAX_IMAGE_DIM):
    """Redimensionar si la imagen es muy grande."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def extract_blocks(gray_array, block_size=BLOCK_SIZE, step=STEP):
    """
    Extraer todos los bloques de la imagen en escala de grises.
    Retorna: lista de (x, y, vector_caracteristicas)
    """
    h, w = gray_array.shape
    blocks = []

    for y in range(0, h - block_size, step):
        for x in range(0, w - block_size, step):
            block = gray_array[y:y+block_size, x:x+block_size].astype(float)

            # Vector de características: DCT simplificada + estadísticas
            mean  = block.mean()
            std   = block.std()
            if std < 2.0:  # bloque uniforme — skip (fondo liso, no útil)
                continue

            # Normalizar
            block_norm = (block - mean) / (std + 1e-8)

            # Tomar 32 valores representativos del bloque (reducción de dimensión)
            flat = block_norm.flatten()
            step_feat = max(1, len(flat) // 32)
            features = flat[::step_feat][:32]

            blocks.append((x, y, features))

    return blocks


def find_copy_move_regions(blocks, sim_threshold=SIM_THRESHOLD, min_distance=MIN_DISTANCE):
    """
    Comparar todos los bloques entre sí para encontrar pares similares
    que estén suficientemente separados (= copias, no el mismo bloque).
    """
    if len(blocks) < 10:
        return []

    positions = np.array([(b[0], b[1]) for b in blocks])
    features  = np.array([b[2] for b in blocks])

    # Normalizar features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_norm = features / norms

    # Calcular similitud coseno entre todos los bloques
    # Para eficiencia, procesar en chunks si hay muchos bloques
    n = len(blocks)
    matches = []

    chunk_size = 200
    for i in range(0, n, chunk_size):
        chunk_feat = features_norm[i:i+chunk_size]
        sims = np.dot(chunk_feat, features_norm.T)  # similitud coseno

        for local_i, global_i in enumerate(range(i, min(i+chunk_size, n))):
            row = sims[local_i]
            # Buscar bloques muy similares
            similar_idx = np.where(row > sim_threshold)[0]

            for j in similar_idx:
                if j <= global_i:  # evitar duplicados
                    continue

                # Calcular distancia espacial
                dist = np.sqrt(
                    (positions[global_i][0] - positions[j][0])**2 +
                    (positions[global_i][1] - positions[j][1])**2
                )

                if dist >= min_distance:
                    matches.append({
                        'source': (int(positions[global_i][0]), int(positions[global_i][1])),
                        'dest':   (int(positions[j][0]),        int(positions[j][1])),
                        'similarity': float(row[j]),
                    })

    return matches


def cluster_matches(matches, block_size=BLOCK_SIZE, min_cluster=3):
    """
    Agrupar matches cercanos en regiones coherentes.
    Un cluster de ≥ min_cluster bloques similares adyacentes = región copiada.
    """
    if not matches:
        return []

    # Agrupar por proximidad espacial (distancia entre fuentes ≤ 2*block_size)
    clusters = []
    used = set()

    for i, m in enumerate(matches):
        if i in used:
            continue
        cluster = [m]
        used.add(i)
        for j, m2 in enumerate(matches):
            if j in used:
                continue
            d = ((m['source'][0]-m2['source'][0])**2 + (m['source'][1]-m2['source'][1])**2)**0.5
            if d <= block_size * 3:
                cluster.append(m2)
                used.add(j)
        if len(cluster) >= min_cluster:
            clusters.append(cluster)

    # Convertir clusters a regiones bounding box
    regions = []
    for cluster in clusters:
        src_xs = [m['source'][0] for m in cluster]
        src_ys = [m['source'][1] for m in cluster]
        dst_xs = [m['dest'][0] for m in cluster]
        dst_ys = [m['dest'][1] for m in cluster]
        avg_sim = sum(m['similarity'] for m in cluster) / len(cluster)

        regions.append({
            'source':      [min(src_xs), min(src_ys), max(src_xs)-min(src_xs)+BLOCK_SIZE, max(src_ys)-min(src_ys)+BLOCK_SIZE],
            'destination': [min(dst_xs), min(dst_ys), max(dst_xs)-min(dst_xs)+BLOCK_SIZE, max(dst_ys)-min(dst_ys)+BLOCK_SIZE],
            'similarity':  round(avg_sim * 100, 1),
            'block_count': len(cluster),
        })

    return regions


def draw_heatmap(img, regions, scale_factor=1.0):
    """Dibujar las regiones detectadas sobre la imagen original."""
    overlay = img.copy().convert('RGBA')
    draw    = ImageDraw.Draw(overlay)

    colors = {
        'source':      (255, 80,  80,  120),  # rojo = región origen (copiada de aquí)
        'destination': (80,  80,  255, 120),  # azul = región destino (pegada aquí)
    }

    for region in regions:
        for rtype in ['source', 'destination']:
            x, y, w, h = [int(v * scale_factor) for v in region[rtype]]
            color = colors[rtype]
            draw.rectangle([x, y, x+w, y+h], fill=color, outline=color[:3]+(230,))

        # Línea conectando origen y destino
        sx = int((region['source'][0] + region['source'][2]//2) * scale_factor)
        sy = int((region['source'][1] + region['source'][3]//2) * scale_factor)
        dx = int((region['destination'][0] + region['destination'][2]//2) * scale_factor)
        dy = int((region['destination'][1] + region['destination'][3]//2) * scale_factor)
        draw.line([sx, sy, dx, dy], fill=(255, 165, 0, 200), width=2)

    # Leyenda
    h_img = overlay.height
    draw.rectangle([0, h_img-30, overlay.width, h_img], fill=(0,0,0,160))
    draw.text((8, h_img-22), f'Copy-Move | {len(regions)} region(es) detectada(s)', fill=(255,255,255,230))

    return overlay.convert('RGB')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'copy-move-detection', 'version': '1.0'})


@app.route('/analyze', methods=['POST'])
def analyze():
    # Validar API key
    key = request.headers.get('X-IVS-Key', '') or request.json.get('api_key', '')
    if key != SECRET_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'image_base64 required'}), 400

    filename = data.get('filename', 'image.jpg')

    try:
        # Decodificar imagen
        img_data = base64.b64decode(data['image_base64'])
        img_orig = Image.open(io.BytesIO(img_data)).convert('RGB')
        orig_size = img_orig.size

        # Redimensionar para procesamiento
        img = resize_if_needed(img_orig)
        proc_size = img.size
        scale = orig_size[0] / proc_size[0]  # factor para mapear coordenadas de vuelta

        # Convertir a escala de grises
        gray = np.array(img.convert('L'))

        # Extraer bloques
        blocks = extract_blocks(gray)

        if len(blocks) < 20:
            return jsonify({
                'detected':       False,
                'confidence':     0,
                'regions':        [],
                'heatmap_base64': None,
                'summary':        'Imagen con muy pocos bloques analizables — no concluyente.',
                'block_count':    len(blocks),
            })

        # Encontrar matches
        matches = find_copy_move_regions(blocks)

        # Agrupar en regiones
        regions = cluster_matches(matches)

        # Determinar confianza
        if not regions:
            confidence = 0
            detected   = False
            summary    = 'No se detectaron regiones copiadas en la imagen.'
        else:
            # La confianza depende del número de bloques en las regiones y su similitud
            avg_blocks = sum(r['block_count'] for r in regions) / len(regions)
            avg_sim    = sum(r['similarity'] for r in regions) / len(regions)
            confidence = min(99, int(avg_sim * 0.7 + min(avg_blocks, 20) * 1.5))
            detected   = confidence >= 60
            summary    = (
                f'Se detectaron {len(regions)} región(es) con patrones de copia. '
                f'Confianza: {confidence}%. '
                f'Las zonas marcadas en rojo (origen) fueron copiadas a las zonas en azul (destino). '
                f'Este patrón es consistente con la manipulación de contenido textual o visual dentro de la imagen.'
                if detected else
                f'Se encontraron {len(regions)} región(es) con cierta similitud, pero la confianza es baja ({confidence}%). '
                'No concluyente — puede deberse a patrones repetitivos naturales.'
            )

        # Generar heatmap solo si se detectó algo
        heatmap_b64 = None
        if detected and regions:
            heatmap_img = draw_heatmap(img_orig, regions, scale_factor=scale)
            buf = io.BytesIO()
            heatmap_img.save(buf, 'JPEG', quality=82)
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'detected':       detected,
            'confidence':     confidence,
            'regions':        regions,
            'heatmap_base64': heatmap_b64,
            'summary':        summary,
            'block_count':    len(blocks),
            'match_count':    len(matches),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'detected': False, 'confidence': 0, 'regions': []}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    print(f'IVS Copy-Move Detection Service v1.0')
    print(f'Escuchando en {host}:{port}')
    print(f'Block size: {BLOCK_SIZE}px | Step: {STEP}px | Threshold: {SIM_THRESHOLD}')
    app.run(host=host, port=port, debug=False)
