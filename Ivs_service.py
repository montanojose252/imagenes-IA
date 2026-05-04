#!/usr/bin/env python3
"""
ivs_service.py — Microservicio Forense IVS v3.0
================================================
Combina dos motores en un solo servicio:

1. Copy-Move Detection
   Detecta regiones copiadas y pegadas dentro de la misma imagen.

2. Google Vision Analysis (cuando GOOGLE_VISION_KEY está configurada)
   - OCR: lee todo el texto visible con coordenadas exactas
   - Object Localization: detecta objetos con bounding boxes
   - Label Detection: etiquetas generales de la escena
   - Cruce de coordenadas con zonas ELA/Copy-Move para identificar
     exactamente QUÉ elemento fue modificado, añadido o eliminado.

Comparación entre dos fotos del mismo lote:
   Si se envían dos imágenes, el sistema compara los elementos
   detectados en ambas y determina qué cambió entre una y otra.

Endpoints:
   GET  /health                → Estado del servicio
   POST /analyze               → Copy-Move Detection
   POST /vision                → Google Vision (OCR + objetos + etiquetas)
   POST /compare               → Comparar dos fotos (qué cambió)

Variables de entorno requeridas:
   IVS_SERVICE_KEY             → Clave de autenticación
   GOOGLE_VISION_KEY           → API Key de Google Cloud Vision (opcional)

Instalación:
   pip install flask pillow numpy scipy requests
"""

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import numpy as np
import base64, io, os, math, requests, json

app = Flask(__name__)

# ── Configuración ─────────────────────────────────────────
BLOCK_SIZE    = 16
STEP          = 8
SIM_THRESHOLD = 0.92
MIN_DISTANCE  = 50
MAX_IMAGE_DIM = 800
SECRET_KEY    = os.environ.get('IVS_SERVICE_KEY',   'ivs_copymove_2026')
VISION_KEY    = os.environ.get('GOOGLE_VISION_KEY', '')

def get_vision_key():
    """Lee la API key en tiempo real en cada llamada."""
    # Primero intentar variable de entorno
    key = os.environ.get('GOOGLE_VISION_KEY', '')
    if key:
        return key
    # Fallback a la global (puede haber sido seteada via /set-key)
    return VISION_KEY

# ── Tipos de objetos en español ───────────────────────────
ETIQUETAS_ES = {
    # Elementos comunes en fotos de trabajo/campo
    'sign':           'cartel / letrero',
    'signage':        'cartel / señalización',
    'banner':         'banner / pancarta',
    'poster':         'póster / afiche',
    'board':          'tablero / pizarra',
    'text':           'texto',
    'number':         'número',
    'date':           'fecha',
    'person':         'persona',
    'face':           'rostro',
    'human face':     'rostro humano',
    'vehicle':        'vehículo',
    'car':            'automóvil',
    'truck':          'camión',
    'license plate':  'placa vehicular',
    'building':       'edificio / estructura',
    'wall':           'pared',
    'floor':          'piso',
    'ceiling':        'techo',
    'door':           'puerta',
    'window':         'ventana',
    'furniture':      'mobiliario',
    'table':          'mesa',
    'chair':          'silla',
    'clothing':       'ropa / vestimenta',
    'uniform':        'uniforme',
    'logo':           'logo / marca',
    'badge':          'credencial / insignia',
    'document':       'documento',
    'paper':          'papel / documento',
    'box':            'caja',
    'product':        'producto',
    'shelf':          'estante',
    'warehouse':      'bodega / almacén',
    'store':          'tienda',
    'office':         'oficina',
    'outdoor':        'exterior',
    'indoor':         'interior',
    'background':     'fondo de la imagen',
    'sky':            'cielo',
    'plant':          'planta',
    'tree':           'árbol',
    'road':           'carretera / camino',
}

def traducir_objeto(nombre_en):
    """Traduce nombre de objeto inglés a español descriptivo."""
    nombre_lower = nombre_en.lower()
    for key, val in ETIQUETAS_ES.items():
        if key in nombre_lower:
            return val
    return nombre_en


# ══════════════════════════════════════════════════════════
# FUNCIONES DE COPY-MOVE
# ══════════════════════════════════════════════════════════

def resize_if_needed(img, max_dim=MAX_IMAGE_DIM):
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def extract_blocks(gray_array, block_size=BLOCK_SIZE, step=STEP):
    h, w = gray_array.shape
    blocks = []
    for y in range(0, h - block_size, step):
        for x in range(0, w - block_size, step):
            block = gray_array[y:y+block_size, x:x+block_size].astype(float)
            mean, std = block.mean(), block.std()
            if std < 2.0:
                continue
            block_norm = (block - mean) / (std + 1e-8)
            flat = block_norm.flatten()
            step_feat = max(1, len(flat) // 32)
            features = flat[::step_feat][:32]
            blocks.append((x, y, features))
    return blocks

def find_copy_move_regions(blocks, sim_threshold=SIM_THRESHOLD, min_distance=MIN_DISTANCE):
    if len(blocks) < 10:
        return []
    positions = np.array([(b[0], b[1]) for b in blocks])
    features  = np.array([b[2] for b in blocks])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_norm = features / norms
    n = len(blocks)
    matches = []
    chunk_size = 200
    for i in range(0, n, chunk_size):
        chunk_feat = features_norm[i:i+chunk_size]
        sims = np.dot(chunk_feat, features_norm.T)
        for local_i, global_i in enumerate(range(i, min(i+chunk_size, n))):
            row = sims[local_i]
            similar_idx = np.where(row > sim_threshold)[0]
            for j in similar_idx:
                if j <= global_i:
                    continue
                dist = math.sqrt(
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
    if not matches:
        return []
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

def draw_copymove_heatmap(img, regions, scale_factor=1.0):
    w_img, h_img = img.size
    result  = img.copy().convert('RGBA')
    overlay = Image.new('RGBA', (w_img, h_img), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    COLOR_ORIGEN  = (220, 38,  38,  140)
    COLOR_DESTINO = (37,  99,  235, 140)
    COLOR_FLECHA  = (251, 191, 36,  230)

    for i, region in enumerate(regions):
        sx, sy, sw, sh = [int(v * scale_factor) for v in region['source']]
        dx, dy, dw, dh = [int(v * scale_factor) for v in region['destination']]
        draw.rectangle([sx, sy, sx+sw, sy+sh], fill=COLOR_ORIGEN)
        draw.rectangle([dx, dy, dx+dw, dy+dh], fill=COLOR_DESTINO)
        for offset in range(3):
            draw.rectangle([sx-offset, sy-offset, sx+sw+offset, sy+sh+offset], outline=(220,38,38,200))
            draw.rectangle([dx-offset, dy-offset, dx+dw+offset, dy+dh+offset], outline=(37,99,235,200))
        cx_src = sx + sw // 2; cy_src = sy + sh // 2
        cx_dst = dx + dw // 2; cy_dst = dy + dh // 2
        draw.line([cx_src, cy_src, cx_dst, cy_dst], fill=COLOR_FLECHA, width=4)
        angle = math.atan2(cy_dst - cy_src, cx_dst - cx_src)
        arrow_len = 18; arrow_angle = math.pi / 6
        ax1 = int(cx_dst - arrow_len * math.cos(angle - arrow_angle))
        ay1 = int(cy_dst - arrow_len * math.sin(angle - arrow_angle))
        ax2 = int(cx_dst - arrow_len * math.cos(angle + arrow_angle))
        ay2 = int(cy_dst - arrow_len * math.sin(angle + arrow_angle))
        draw.polygon([(cx_dst, cy_dst), (ax1, ay1), (ax2, ay2)], fill=COLOR_FLECHA)
        num = str(i + 1)
        draw.rectangle([sx+4-2, sy+4-2, sx+4+16, sy+4+16], fill=(0,0,0,180))
        draw.text((sx+4, sy+4), num, fill=(255,255,255,255))
        draw.rectangle([dx+4-2, dy+4-2, dx+4+16, dy+4+16], fill=(0,0,0,180))
        draw.text((dx+4, dy+4), num, fill=(255,255,255,255))

    result = Image.alpha_composite(result, overlay).convert('RGB')
    draw_f = ImageDraw.Draw(result)
    ley_h = 90; ley_y = h_img - ley_h
    ley_bg = Image.new('RGB', (w_img, ley_h), (15, 23, 42))
    result.paste(ley_bg, (0, ley_y))
    draw_f = ImageDraw.Draw(result)
    draw_f.text((12, ley_y+6),  f'ANÁLISIS FORENSE — {len(regions)} ZONA(S) CON COPIA-PEGA DETECTADA', fill=(251,191,36))
    items = [
        ((220,38,38),  'ZONA ORIGEN: de aquí se copió'),
        ((37,99,235),  'ZONA DESTINO: aquí se pegó para ocultar'),
        ((251,191,36), 'FLECHA: dirección de la manipulación'),
    ]
    for j, (color, texto) in enumerate(items):
        ix = 12 + j * (w_img // 3); iy = ley_y + 32
        draw_f.rectangle([ix, iy, ix+14, iy+14], fill=color)
        draw_f.text((ix+20, iy), texto, fill=(200,200,200))
    return result


# ══════════════════════════════════════════════════════════
# FUNCIONES DE GOOGLE VISION
# ══════════════════════════════════════════════════════════

def call_vision_api(image_b64, features):
    """Llama a Google Cloud Vision API."""
    vision_key = get_vision_key()
    if not vision_key:
        return None, 'GOOGLE_VISION_KEY no configurada'
    url = f'https://vision.googleapis.com/v1/images:annotate?key={vision_key}'
    body = {
        'requests': [{
            'image': {'content': image_b64},
            'features': features,
            'imageContext': {'languageHints': ['es', 'en']}
        }]
    }
    try:
        resp = requests.post(url, json=body, timeout=15)
        if resp.status_code != 200:
            return None, f'Vision API error HTTP {resp.status_code}: {resp.text[:200]}'
        data = resp.json()
        if 'error' in data:
            return None, data['error'].get('message', 'Unknown error')
        return data['responses'][0], None
    except Exception as e:
        return None, str(e)


def bbox_to_rect(vertices):
    """Convierte vértices de Vision API a [x, y, w, h]."""
    xs = [v.get('x', 0) for v in vertices]
    ys = [v.get('y', 0) for v in vertices]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs)-x0, max(ys)-y0]


def normalize_bbox(bbox, img_w, img_h):
    """Normaliza bounding box a porcentajes para comparación entre imágenes."""
    x, y, w, h = bbox
    return {
        'x_pct': round(x / img_w * 100, 1),
        'y_pct': round(y / img_h * 100, 1),
        'w_pct': round(w / img_w * 100, 1),
        'h_pct': round(h / img_h * 100, 1),
    }


def zones_overlap(r1, r2, threshold=0.3):
    """
    Determina si dos bounding boxes [x,y,w,h] se solapan significativamente.
    threshold: fracción de solapamiento requerida (0.3 = 30%)
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    inter_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    inter_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter_area = inter_x * inter_y
    area1 = w1 * h1
    area2 = w2 * h2
    if area1 == 0 or area2 == 0:
        return False
    overlap = inter_area / min(area1, area2)
    return overlap >= threshold


def analyze_vision_full(image_b64):
    """
    Ejecuta análisis completo de Google Vision:
    - TEXT_DETECTION: todo el texto con coordenadas
    - OBJECT_LOCALIZATION: objetos con bounding boxes
    - LABEL_DETECTION: etiquetas generales de la escena
    """
    features = [
        {'type': 'TEXT_DETECTION',       'maxResults': 50},
        {'type': 'OBJECT_LOCALIZATION',  'maxResults': 20},
        {'type': 'LABEL_DETECTION',      'maxResults': 15},
    ]
    response, error = call_vision_api(image_b64, features)
    if error:
        return None, error

    result = {
        'textos':   [],   # texto con coordenadas
        'objetos':  [],   # objetos localizados
        'etiquetas': [],  # etiquetas generales
    }

    # ── Texto detectado ───────────────────────────────────
    full_text = ''
    if 'textAnnotations' in response:
        annots = response['textAnnotations']
        # El primer elemento es el texto completo de la imagen
        if annots:
            full_text = annots[0].get('description', '').strip()
        # Los siguientes son palabras/frases individuales con coordenadas
        for ann in annots[1:]:
            txt = ann.get('description', '').strip()
            if not txt or len(txt) < 2:
                continue
            verts = ann.get('boundingPoly', {}).get('vertices', [])
            if verts:
                result['textos'].append({
                    'texto':    txt,
                    'bbox':     bbox_to_rect(verts),
                    'tipo':     clasificar_texto(txt),
                })

    result['texto_completo'] = full_text

    # ── Objetos localizados ───────────────────────────────
    if 'localizedObjectAnnotations' in response:
        img_data = base64.b64decode(image_b64)
        img_tmp  = Image.open(io.BytesIO(img_data))
        img_w, img_h = img_tmp.size

        for obj in response['localizedObjectAnnotations']:
            nombre_en  = obj.get('name', '')
            confianza  = obj.get('score', 0)
            if confianza < 0.5:
                continue
            # Convertir vértices normalizados a píxeles
            verts_norm = obj.get('boundingPoly', {}).get('normalizedVertices', [])
            if not verts_norm:
                continue
            xs = [v.get('x', 0) * img_w for v in verts_norm]
            ys = [v.get('y', 0) * img_h for v in verts_norm]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))]
            result['objetos'].append({
                'nombre_en': nombre_en,
                'nombre_es': traducir_objeto(nombre_en),
                'confianza': round(confianza * 100, 1),
                'bbox':      bbox,
            })

    # ── Etiquetas generales ───────────────────────────────
    if 'labelAnnotations' in response:
        for lbl in response['labelAnnotations']:
            conf = lbl.get('score', 0)
            if conf < 0.6:
                continue
            desc = lbl.get('description', '')
            result['etiquetas'].append({
                'nombre_en': desc,
                'nombre_es': traducir_objeto(desc),
                'confianza': round(conf * 100, 1),
            })

    return result, None


def clasificar_texto(texto):
    """Clasifica el tipo de texto detectado."""
    import re
    # Fecha (DD-MM-YY, MM/DD/YYYY, etc.)
    if re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', texto):
        return 'fecha'
    # Solo números
    if re.match(r'^[\d\s\-\.]+$', texto):
        return 'número'
    # Hora
    if re.search(r'\d{1,2}:\d{2}', texto):
        return 'hora'
    # Texto largo → etiqueta/cartel
    if len(texto) > 15:
        return 'texto largo'
    return 'texto'


def cruzar_con_zonas_ela(vision_result, ela_zones):
    """
    Cruza los elementos detectados por Vision con las zonas
    marcadas por ELA como editadas.

    ela_zones: lista de {'zone': 'Middle-center', 'bbox': [x,y,w,h], 'nivel': 'HIGH'}
    Retorna lista de elementos que están dentro de zonas editadas.
    """
    if not ela_zones or not vision_result:
        return []

    elementos_en_zona = []

    for zona in ela_zones:
        zona_bbox = zona.get('bbox')
        if not zona_bbox:
            continue
        nivel = zona.get('nivel', 'LOW')

        # Verificar textos en la zona editada
        for txt in vision_result.get('textos', []):
            if zones_overlap(zona_bbox, txt['bbox'], threshold=0.25):
                elementos_en_zona.append({
                    'tipo':        'texto',
                    'elemento':    f'Texto "{txt["texto"]}"',
                    'tipo_texto':  txt['tipo'],
                    'bbox':        txt['bbox'],
                    'zona_ela':    zona.get('zone'),
                    'nivel_ela':   nivel,
                    'descripcion': f'Se detectó texto "{txt["texto"]}" ({txt["tipo"]}) en la zona marcada como editada ({zona.get("zone")})',
                })

        # Verificar objetos en la zona editada
        for obj in vision_result.get('objetos', []):
            if zones_overlap(zona_bbox, obj['bbox'], threshold=0.25):
                elementos_en_zona.append({
                    'tipo':        'objeto',
                    'elemento':    obj['nombre_es'],
                    'confianza':   obj['confianza'],
                    'bbox':        obj['bbox'],
                    'zona_ela':    zona.get('zone'),
                    'nivel_ela':   nivel,
                    'descripcion': f'Se detectó "{obj["nombre_es"]}" (confianza {obj["confianza"]}%) en la zona marcada como editada ({zona.get("zone")})',
                })

    # Deduplicar por elemento
    vistos = set()
    resultado = []
    for e in elementos_en_zona:
        key = e['elemento']
        if key not in vistos:
            vistos.add(key)
            resultado.append(e)

    return resultado


def comparar_dos_fotos(vision_a, vision_b, img_w, img_h):
    """
    Compara los elementos detectados en dos fotos del mismo lote.
    Identifica qué fue añadido, eliminado o modificado entre A y B.
    """
    cambios = []

    # ── Comparar textos ───────────────────────────────────
    textos_a = {t['texto']: t for t in vision_a.get('textos', [])}
    textos_b = {t['texto']: t for t in vision_b.get('textos', [])}

    # Textos en A pero no en B → eliminados
    for txt, data in textos_a.items():
        if txt not in textos_b:
            # Verificar si hay un texto diferente en la misma posición
            texto_reemplazado = None
            for txt_b, data_b in textos_b.items():
                if zones_overlap(data['bbox'], data_b['bbox'], threshold=0.5):
                    texto_reemplazado = txt_b
                    break

            if texto_reemplazado:
                cambios.append({
                    'tipo':        'modificacion',
                    'elemento':    f'texto "{txt}" → "{texto_reemplazado}"',
                    'descripcion': f'El texto "{txt}" fue reemplazado por "{texto_reemplazado}" en la misma zona de la imagen.',
                    'bbox_a':      data['bbox'],
                    'bbox_b':      textos_b[texto_reemplazado]['bbox'],
                    'tipo_texto':  data.get('tipo', 'texto'),
                    'gravedad':    'alta' if data.get('tipo') == 'fecha' else 'media',
                })
            else:
                cambios.append({
                    'tipo':        'eliminacion',
                    'elemento':    f'texto "{txt}"',
                    'descripcion': f'El texto "{txt}" presente en la imagen original no aparece en la imagen modificada.',
                    'bbox_a':      data['bbox'],
                    'bbox_b':      None,
                    'tipo_texto':  data.get('tipo', 'texto'),
                    'gravedad':    'alta' if data.get('tipo') == 'fecha' else 'media',
                })

    # Textos en B pero no en A → añadidos
    for txt, data in textos_b.items():
        if txt not in textos_a:
            # Verificar si reemplaza algo (ya manejado arriba)
            es_reemplazo = any(
                c.get('elemento','').endswith(f'→ "{txt}"')
                for c in cambios
            )
            if not es_reemplazo:
                # Verificar si hay texto de A en la misma posición
                tiene_origen = any(
                    zones_overlap(data_a['bbox'], data['bbox'], threshold=0.5)
                    for data_a in textos_a.values()
                )
                if not tiene_origen:
                    cambios.append({
                        'tipo':        'adicion',
                        'elemento':    f'texto "{txt}"',
                        'descripcion': f'Se añadió el texto "{txt}" que no existía en la imagen original.',
                        'bbox_a':      None,
                        'bbox_b':      data['bbox'],
                        'tipo_texto':  data.get('tipo', 'texto'),
                        'gravedad':    'alta' if data.get('tipo') == 'fecha' else 'media',
                    })

    # ── Comparar objetos ──────────────────────────────────
    objetos_a = {o['nombre_en']: o for o in vision_a.get('objetos', [])}
    objetos_b = {o['nombre_en']: o for o in vision_b.get('objetos', [])}

    # Objetos en A pero no en B → eliminados
    for nombre, data in objetos_a.items():
        if nombre not in objetos_b:
            cambios.append({
                'tipo':        'eliminacion_objeto',
                'elemento':    data['nombre_es'],
                'descripcion': f'El elemento "{data["nombre_es"]}" presente en la imagen original no se detecta en la imagen modificada.',
                'bbox_a':      data['bbox'],
                'bbox_b':      None,
                'gravedad':    'media',
            })

    # Objetos en B pero no en A → añadidos
    for nombre, data in objetos_b.items():
        if nombre not in objetos_a:
            cambios.append({
                'tipo':        'adicion_objeto',
                'elemento':    data['nombre_es'],
                'descripcion': f'Se detectó un elemento nuevo "{data["nombre_es"]}" en la imagen modificada que no existía en la original.',
                'bbox_b':      data['bbox'],
                'bbox_a':      None,
                'gravedad':    'media',
            })

    # ── Comparar etiquetas de escena ──────────────────────
    etiq_a = {e['nombre_en'] for e in vision_a.get('etiquetas', [])}
    etiq_b = {e['nombre_en'] for e in vision_b.get('etiquetas', [])}
    nuevas   = etiq_b - etiq_a
    perdidas = etiq_a - etiq_b

    if nuevas or perdidas:
        desc_partes = []
        if nuevas:
            desc_partes.append('elementos nuevos en escena: ' + ', '.join(traducir_objeto(e) for e in nuevas))
        if perdidas:
            desc_partes.append('elementos que desaparecieron: ' + ', '.join(traducir_objeto(e) for e in perdidas))
        if desc_partes:
            cambios.append({
                'tipo':        'cambio_escena',
                'elemento':    'composición de la escena',
                'descripcion': 'Cambios en la composición general: ' + '. '.join(desc_partes) + '.',
                'gravedad':    'baja',
            })

    return cambios


def generar_resumen_vision(vision_result, elementos_editados, cambios_entre_fotos=None):
    """
    Genera un resumen en lenguaje natural de los hallazgos de Vision.
    """
    partes = []

    # Resumen de contenido
    if vision_result.get('texto_completo'):
        textos_importantes = [
            t for t in vision_result.get('textos', [])
            if t.get('tipo') in ('fecha', 'número', 'texto largo')
        ]
        if textos_importantes:
            textos_str = ', '.join(f'"{t["texto"]}"' for t in textos_importantes[:5])
            partes.append(f'Texto visible en la imagen: {textos_str}.')

    objetos_principales = [
        o['nombre_es'] for o in vision_result.get('objetos', [])[:5]
    ]
    if objetos_principales:
        partes.append(f'Elementos detectados: {", ".join(objetos_principales)}.')

    # Elementos en zonas editadas
    if elementos_editados:
        for elem in elementos_editados[:3]:
            partes.append(f'⚠ {elem["descripcion"]}')

    # Cambios entre fotos
    if cambios_entre_fotos:
        for cambio in cambios_entre_fotos[:5]:
            gravedad_icon = '🔴' if cambio['gravedad'] == 'alta' else '🟠'
            partes.append(f'{gravedad_icon} {cambio["descripcion"]}')

    return ' '.join(partes) if partes else 'Análisis de contenido completado.'


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

def validate_key(req):
    key = req.headers.get('X-IVS-Key', '') or (req.get_json() or {}).get('api_key', '')
    return key == SECRET_KEY


@app.route('/set-key', methods=['POST'])
def set_key():
    """
    Configura la GOOGLE_VISION_KEY en tiempo de ejecución.
    Útil cuando Railway no inyecta las variables correctamente.
    Body: {"api_key": "AIzaSy...", "admin_key": "ivs_copymove_2026"}
    """
    data = request.get_json() or {}
    admin = data.get('admin_key', '')
    if admin != SECRET_KEY and admin != 'ivs_copymove_2026':
        return jsonify({'error': 'Unauthorized'}), 401

    new_key = data.get('api_key', '').strip()
    if not new_key:
        return jsonify({'error': 'api_key requerida'}), 400

    global VISION_KEY
    VISION_KEY = new_key
    os.environ['GOOGLE_VISION_KEY'] = new_key

    return jsonify({
        'status':  'ok',
        'message': 'GOOGLE_VISION_KEY configurada correctamente',
        'longitud': len(new_key),
        'inicio':   new_key[:8],
    })


@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint de diagnóstico — muestra qué variables están disponibles."""
    vision_raw  = os.environ.get('GOOGLE_VISION_KEY', 'NO_ENCONTRADA')
    # Mostrar todas las variables de entorno (solo nombres, no valores)
    todas_vars  = list(os.environ.keys())
    railway_vars = [k for k in todas_vars if k.startswith('RAILWAY')]
    custom_vars  = [k for k in todas_vars if not k.startswith('RAILWAY') and k not in ('PATH','HOME','USER','SHELL','LANG','LC_ALL','PWD','SHLVL','_')]
    return jsonify({
        'GOOGLE_VISION_KEY_presente': bool(vision_raw and vision_raw != 'NO_ENCONTRADA'),
        'GOOGLE_VISION_KEY_longitud': len(vision_raw) if vision_raw != 'NO_ENCONTRADA' else 0,
        'VISION_KEY_global':          bool(VISION_KEY),
        'railway_vars_disponibles':   railway_vars,
        'custom_vars_disponibles':    custom_vars,
        'total_vars_entorno':         len(todas_vars),
    })

@app.route('/health', methods=['GET'])
def health():
    vision_key_live = get_vision_key()
    global VISION_KEY
    VISION_KEY = vision_key_live
    return jsonify({
        'status':        'ok',
        'service':       'IVS Forensic Service',
        'version':       '2.0',
        'copy_move':     True,
        'google_vision': bool(vision_key_live),
        'endpoints':     ['/health', '/analyze', '/vision', '/compare', '/set-key', '/debug'],
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Copy-Move Detection — igual que antes para compatibilidad."""
    if not validate_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'image_base64 required'}), 400

    try:
        img_data = base64.b64decode(data['image_base64'])
        img_orig = Image.open(io.BytesIO(img_data)).convert('RGB')
        orig_size = img_orig.size
        img = resize_if_needed(img_orig)
        scale = orig_size[0] / img.size[0]
        gray = np.array(img.convert('L'))
        blocks = extract_blocks(gray)

        if len(blocks) < 20:
            return jsonify({
                'detected': False, 'confidence': 0, 'regions': [],
                'heatmap_base64': None,
                'summary': 'Imagen con muy pocos bloques analizables.',
                'block_count': len(blocks),
            })

        matches = find_copy_move_regions(blocks)
        regions = cluster_matches(matches)

        if not regions:
            return jsonify({
                'detected': False, 'confidence': 0, 'regions': regions,
                'heatmap_base64': None,
                'summary': 'No se detectaron regiones copiadas en la imagen.',
                'block_count': len(blocks), 'match_count': len(matches),
            })

        avg_blocks = sum(r['block_count'] for r in regions) / len(regions)
        avg_sim    = sum(r['similarity'] for r in regions) / len(regions)
        confidence = min(99, int(avg_sim * 0.7 + min(avg_blocks, 20) * 1.5))
        detected   = confidence >= 60

        heatmap_b64 = None
        if detected:
            heatmap_img = draw_copymove_heatmap(img_orig, regions, scale_factor=scale)
            buf = io.BytesIO()
            heatmap_img.save(buf, 'JPEG', quality=85)
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

        summary = (
            f'Se detectaron {len(regions)} región(es) con patrones de copia-pega. '
            f'Confianza: {confidence}%. Las zonas marcadas en rojo son el origen '
            f'(de donde se copió) y las azules el destino (donde se pegó). '
            f'Esta técnica se usa para ocultar fechas, carteles, placas u otros elementos identificadores.'
            if detected else
            f'Se encontraron {len(regions)} región(es) con cierta similitud, pero la confianza es baja ({confidence}%). No concluyente.'
        )

        return jsonify({
            'detected': detected, 'confidence': confidence,
            'regions': regions, 'heatmap_base64': heatmap_b64,
            'summary': summary, 'block_count': len(blocks), 'match_count': len(matches),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'detected': False, 'confidence': 0, 'regions': []}), 500


@app.route('/vision', methods=['POST'])
def vision():
    """
    Google Vision Analysis:
    - OCR completo con coordenadas
    - Detección de objetos con bounding boxes
    - Etiquetas de la escena
    - Cruce con zonas ELA para identificar qué elemento fue editado
    """
    if not validate_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    if not get_vision_key():
        return jsonify({'error': 'GOOGLE_VISION_KEY no configurada en Railway Variables'}), 503

    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'image_base64 required'}), 400

    # Zonas ELA opcionales para cruce
    ela_zones = data.get('ela_zones', [])

    try:
        vision_result, error = analyze_vision_full(data['image_base64'])
        if error:
            return jsonify({'error': error}), 500

        # Cruzar con zonas ELA si se proporcionaron
        elementos_editados = []
        if ela_zones:
            elementos_editados = cruzar_con_zonas_ela(vision_result, ela_zones)

        resumen = generar_resumen_vision(vision_result, elementos_editados)

        return jsonify({
            'texto_completo':    vision_result.get('texto_completo', ''),
            'textos':            vision_result.get('textos', []),
            'objetos':           vision_result.get('objetos', []),
            'etiquetas':         vision_result.get('etiquetas', []),
            'elementos_editados': elementos_editados,
            'resumen':           resumen,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """
    Comparar dos fotos del mismo lote.
    Identifica exactamente qué cambió entre imagen A e imagen B:
    - Texto modificado (ej: fecha cambiada)
    - Objetos añadidos o eliminados
    - Cambios en la composición de la escena

    Body: {
        "image_a_base64": "...",
        "image_b_base64": "...",
        "ela_zones_a": [...],   // opcional
        "ela_zones_b": [...]    // opcional
    }
    """
    if not validate_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    if not get_vision_key():
        return jsonify({'error': 'GOOGLE_VISION_KEY no configurada en Railway Variables'}), 503

    data = request.get_json()
    if not data or 'image_a_base64' not in data or 'image_b_base64' not in data:
        return jsonify({'error': 'image_a_base64 y image_b_base64 requeridos'}), 400

    try:
        # Analizar ambas imágenes
        vision_a, err_a = analyze_vision_full(data['image_a_base64'])
        if err_a:
            return jsonify({'error': f'Error en imagen A: {err_a}'}), 500

        vision_b, err_b = analyze_vision_full(data['image_b_base64'])
        if err_b:
            return jsonify({'error': f'Error en imagen B: {err_b}'}), 500

        # Dimensiones de imagen A para normalización
        img_data = base64.b64decode(data['image_a_base64'])
        img_tmp  = Image.open(io.BytesIO(img_data))
        img_w, img_h = img_tmp.size

        # Comparar
        cambios = comparar_dos_fotos(vision_a, vision_b, img_w, img_h)

        # Elementos en zonas editadas de cada imagen
        elem_a = cruzar_con_zonas_ela(vision_a, data.get('ela_zones_a', []))
        elem_b = cruzar_con_zonas_ela(vision_b, data.get('ela_zones_b', []))

        # Resumen narrativo
        resumen_partes = []
        cambios_altos = [c for c in cambios if c.get('gravedad') == 'alta']
        cambios_medios = [c for c in cambios if c.get('gravedad') == 'media']

        if cambios_altos:
            for c in cambios_altos:
                resumen_partes.append(c['descripcion'])
        if cambios_medios:
            for c in cambios_medios[:2]:
                resumen_partes.append(c['descripcion'])
        if not cambios:
            resumen_partes.append('No se detectaron diferencias significativas entre las dos imágenes.')

        # Determinar tipo de manipulación principal
        tipos = [c['tipo'] for c in cambios]
        if 'modificacion' in tipos:
            tipo_principal = 'modificacion_contenido'
        elif 'adicion' in tipos or 'adicion_objeto' in tipos:
            tipo_principal = 'adicion_elementos'
        elif 'eliminacion' in tipos or 'eliminacion_objeto' in tipos:
            tipo_principal = 'eliminacion_elementos'
        else:
            tipo_principal = 'sin_cambios_detectados'

        etiquetas_descripcion = {
            'modificacion_contenido':  'Modificación de contenido — texto o elementos reemplazados',
            'adicion_elementos':       'Adición de elementos — se añadió contenido a la imagen',
            'eliminacion_elementos':   'Eliminación de elementos — se quitó contenido de la imagen',
            'sin_cambios_detectados':  'Sin cambios significativos detectados entre las dos imágenes',
        }

        return jsonify({
            'cambios':           cambios,
            'tipo_principal':    tipo_principal,
            'tipo_descripcion':  etiquetas_descripcion[tipo_principal],
            'total_cambios':     len(cambios),
            'cambios_criticos':  len(cambios_altos),
            'resumen':           ' '.join(resumen_partes),
            'vision_a':          vision_a,
            'vision_b':          vision_b,
            'elementos_zona_a':  elem_a,
            'elementos_zona_b':  elem_b,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    print(f'IVS Forensic Service v2.0')
    print(f'Copy-Move: ✓  |  Google Vision: {"✓" if VISION_KEY else "✗ (configurar GOOGLE_VISION_KEY)"}')
    print(f'Escuchando en {host}:{port}')
    app.run(host=host, port=port, debug=False)
