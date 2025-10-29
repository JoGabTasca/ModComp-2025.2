## ModComp-2025.2

Aplicação Django com calculadoras de métodos numéricos (bisseção, Newton, Gauss).

### Alunos
- Ewerton
- Joao Gabriel Tasca
- Marcos Paulo
- Miguel Veiga
- Otto

---

### Requisitos
- Python 3.11+
- pip

Opcional (recomendado): virtualenv (venv)

### Como rodar (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

### Estrutura do projeto
```
core/                # Configurações do projeto Django
calculo/             # App com views, templates e métodos numéricos
  templates/calculo/ # Páginas HTML
  static/            # Assets estáticos
requirements.txt     # Dependências
manage.py            # Entrypoint Django
```