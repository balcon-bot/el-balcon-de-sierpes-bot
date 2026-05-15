import random

class StyleEngine:
    def __init__(self):
        self.vocabulary = {
            "procesión": "Estación de Penitencia",
            "sale": "realiza su Estación de Penitencia",
            "iglesia": "Templo Sacramental",
            "pasos": "Misterios",
            "cristo": "Santo Cristo",
            "virgen": "Santa María",
            "gente": "Pueblo fiel",
            "lluvia": "agua bendita",
            "calles": "recorrido oficial",
            "música": "agrupación musical"
        }
        self.openers = ["📜 Crónica de la Devoción:", "🕯️ Apuntes del Mayordomo:", "⚜️ Desde la Capilla:", "📰 Actualidad Cofrade:"]
        self.closers = ["Viva Sevilla.", "Fe y Devoción.", "🕯️", "Por la senda del sentimiento."]

    def generate_post_content(self, raw_news):
        opener = random.choice(self.openers)
        closer = random.choice(self.closers)
        summary = raw_news['summary']
        for key, value in self.vocabulary.items():
            summary = summary.replace(key, value)
        text = f"{opener}

{raw_news['title'].upper()}

{summary}

{closer}"
        if len(text) > 240:
            text = text[:237] + "..."
        return text
