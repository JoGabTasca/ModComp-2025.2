from django.shortcuts import render
import sympy
from sympy.core.expr import Expr
from .bissecao_method import metodo_bissecao
from .newton_method import newton_raphson
from .gauss_method import gauss_somente_web, resolver_por_svd_web, resolver_por_minimos_quadrados_web
import numpy as np 
import re 



# --- View da Página Inicial ---
def home_calculo_view(request):
    """
    View para a página inicial do app 'calculo', onde o usuário escolhe o método.
    """
    return render(request, 'calculo/home_calculo.html')


# --- View da Calculadora de Newton ---
def newton_calculator_view(request):
    context = {
        'form_data': {      # Valores padrão para o formulário na primeira carga
            'funcao_str': 'x**2 - 4',
            'x0_str': '1.0',
            'erro_str': '1e-7',
            'max_iter_str': '100',
        }
    }

    derivada_calculada_str = ""     # Variável para mostrar a derivada no HTML 
    f_na_raiz = None        # Variável para mostrar f(raiz) no HTML

    if request.method == 'POST':
        funcao_str = request.POST.get('funcao_str', '').strip().lower()
        x0_str = request.POST.get('x0_str', '').strip()
        erro_str = request.POST.get('erro_str', '').strip()
        max_iter_str = request.POST.get('max_iter_str', '100').strip()

        # --- DEBUGGING ---
        print("-" * 40)
        print(f"DEBUG DJANGO VIEW: funcao_str recebida: '{funcao_str}'")
        print(f"DEBUG DJANGO VIEW: tipo de funcao_str: {type(funcao_str)}")
        print(f"DEBUG DJANGO VIEW: x0_str: '{x0_str}', erro_str: '{erro_str}', max_iter_str: '{max_iter_str}'")
        print("-" * 40)
        # --- FIM DEBUGGING ---

        context['form_data'] = {        # Atualiza com os dados enviados
            'funcao_str': funcao_str,
            'x0_str': x0_str,
            'erro_str': erro_str,
            'max_iter_str': max_iter_str,
        }

        try:
            # --- VALIDAÇÃO E CONVERSÃO DOS INPUTS NUMÉRICOS ---
            if not x0_str or not erro_str or not max_iter_str:
                raise ValueError("Todos os campos numéricos (x0, tolerância, máx. iterações) são obrigatórios.")
            
            x0 = float(x0_str.replace(',', '.'))
            erro = float(erro_str.replace(',', '.'))
            max_iter = int(max_iter_str)

            if erro <= 0:
                raise ValueError("A tolerância deve ser um valor positivo.")
            if max_iter <= 0:
                raise ValueError("O número máximo de iterações deve ser positivo.")
            # --- FIM DA VALIDAÇÃO ---

        # --- TRATAMENTO DE ERROS E RECARREGAMENTO DA PÁGINA ---
        except ValueError as e:
            context['erro_input'] = f"Erro nos valores numéricos: {e} Verifique se usou ponto '.' como separador decimal ou se os valores são válidos."
            return render(request, 'calculo/newton_calculator.html', context)

        try:
            # --- DEBUGGING ---
            print("DEBUG DJANGO VIEW: Entrando no bloco try do SymPy...")
            # --- DEBUGGING ---
            x_sym = sympy.symbols('x')
            
            # --- CUIDADOS COM SEGURANÇA NO INPUT DO USUÁRIO ---
            allowed_functions = {
                "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                "exp": sympy.exp, "ln": sympy.log, "log": sympy.log,
                "log10": lambda arg: sympy.log(arg, 10),
                "sqrt": sympy.sqrt, "abs": sympy.Abs, "fabs": sympy.Abs,
                "pi": sympy.pi, "e": sympy.E,
                "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
                "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
            }
            
            local_scope = allowed_functions.copy()      # Cria o "escopo local" seguro para o 'sympify'
            local_scope['x'] = x_sym

            if not funcao_str:
                raise ValueError("A expressão da função não pode estar vazia.")
            
            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: Antes do sympify, funcao_str: '{funcao_str}'")
            # --- DEBUGGING ---

            func_sympy = sympy.sympify(funcao_str, locals=local_scope)

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: Após sympify, func_sympy: {func_sympy}")
            # --- DEBUGGING ---
            
            # --- VALIDAÇÕES SOBRE A EXPRESSÃO MATEMÁTICA ---
            if not isinstance(func_sympy, Expr):
                raise ValueError(f"A função '{funcao_str}' não foi interpretada como uma expressão matemática escalar válida. Verifique a sintaxe.")

            if func_sympy.is_number:
                if sympy.Eq(func_sympy, 0):
                    raise ValueError("A função fornecida é '0'. Não é possível aplicar Newton-Raphson.")
                else:
                    raise ValueError(f"A função fornecida é uma constante '{func_sympy}'. Não há raízes (a menos que a constante seja 0).")

            # A verificação if x_sym not in func_sympy.free_symbols: foi removida por enquanto
            # pois a lógica de derivada zero e is_number deve cobrir os casos problemáticos.


            # --- LÓGICA DE NEWTON-RAPHSON ---
            derivada_sympy = sympy.diff(func_sympy, x_sym)      # Calcula a derivada de f(x) em relação a 'x'

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: Após diff, derivada_sympy: {derivada_sympy}")
            # --- DEBUGGING ---

            derivada_calculada_str = str(derivada_sympy)        # Salva a string para mostrar no HTML

            func_callable = sympy.lambdify(x_sym, func_sympy, modules=['math'])

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: func_callable criada: {func_callable}")
            # --- DEBUGGING ---

            derivada_callable = sympy.lambdify(x_sym, derivada_sympy, modules=['math'])

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: derivada_callable criada: {derivada_callable}")
            # --- DEBUGGING ---

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: Antes de chamar newton_raphson com x0={x0}, tol={erro}")
            # --- DEBUGGING ---
            
            # --- CÁLCULO DO MÉTODO DE NEWTON-RAPHSON ---
            raiz, iteracoes, f_na_raiz, atingiu_max_iter, erro_calculado = newton_raphson(
                func_callable,
                derivada_callable,
                x0,
                erro,
                max_iter
            )

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: Após newton_raphson: raiz={raiz}, iter={iteracoes}")
            # --- DEBUGGING ---
            
            # --- MONTA A MENSAGEM BASEADA NO CRITÉRIO DE PARADA ---
            if raiz is None:
                mensagem = "Falha: Derivada igual a zero."
            elif atingiu_max_iter:
                mensagem = "Máximo de iterações atingido."
            else:
                mensagem = "Convergiu pelo erro relativo."
            
            context['resultado'] = {
                'raiz': raiz, 
                'iteracoes': iteracoes, 
                'f_na_raiz': f_na_raiz,
                'erro_calculado': erro_calculado,
                'mensagem': mensagem,
            }
            context['derivada_calculada_str'] = derivada_calculada_str


        # --- CAPTURA DE ERROS ---
        except (sympy.SympifyError, TypeError, NameError) as e:     # Captura erros do SymPy

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (SympifyError, TypeError, NameError): {type(e).__name__} - {e}")
            # --- DEBUGGING ---

            context['erro_sympy'] = f"Erro ao processar a função: '{e}'. Verifique a sintaxe. Use 'x' como variável e funções como sin(x), exp(x), log(x), etc."

        except ValueError as e:     # Captura erros de validação

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (ValueError): {e}")
            # --- DEBUGGING ---

            context['erro_sympy'] = str(e)

        except Exception as e:      # Captura qualquer outro erro inesperado

            # --- DEBUGGING ---
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (Outra Exceção): {type(e).__name__} - {e}")
            # --- DEBUGGING ---

            context['erro_sympy'] = f"Ocorreu um erro inesperado no processamento da função: {e}"

    # Renderiza a página
    # Se for GET, renderiza com os valores padrão
    # Se for POST, renderiza com os valores enviados e com o 'resultado' ou 'erro_sympy'
    return render(request, 'calculo/newton_calculator.html', context)


# --- View da Calculadora de Bissecção ---
def bissecao_calculator_view(request):
    context = {
        'form_data': {      # Valores padrão para o formulário na primeira carga
            'funcao_str': 'x**3 - x - 2',
            'a_str': '1.0',
            'b_str': '2.0',
            'erro_str': '1e-5',
            'max_iter_str': '100',
        }
    }

    f_na_raiz = None        # Para armazenar o valor de f(raiz_encontrada)

    if request.method == 'POST':
        funcao_str = request.POST.get('funcao_str', '').strip().lower()
        a_str = request.POST.get('a_str', '').strip()
        b_str = request.POST.get('b_str', '').strip()
        erro_str = request.POST.get('erro_str', '').strip()
        max_iter_str = request.POST.get('max_iter_str', '100').strip()

        context['form_data'] = {
            'funcao_str': funcao_str,
            'a_str': a_str,
            'b_str': b_str,
            'erro_str': erro_str,
            'max_iter_str': max_iter_str,
        }

        # --- VALIDAÇÃO DE INPUTS NUMÉRICOS ---
        try:
            if not a_str or not b_str or not erro_str or not max_iter_str:
                raise ValueError("Todos os campos numéricos (a, b, tolerância, máx. iterações) são obrigatórios.")
            
            val_a = float(a_str.replace(',', '.'))
            val_b = float(b_str.replace(',', '.'))
            erro = float(erro_str.replace(',', '.'))
            max_iter = int(max_iter_str)

            # --- VALIDAÇÕES ESPECÍFICAS DO MÉTODO DA BISSEÇÃO ---
            if erro <= 0:
                raise ValueError("A tolerância deve ser um valor positivo.")
            if max_iter <= 0:
                raise ValueError("O número máximo de iterações deve ser positivo.")
            if val_a >= val_b:
                raise ValueError("O valor de 'a' deve ser menor que o valor de 'b'.")

        except ValueError as e:
            context['erro_input'] = f"Erro nos valores numéricos: {e}"
            return render(request, 'calculo/bissecao_calculator.html', context)

        try:
            x_sym = sympy.symbols('x')
            allowed_functions = {       # Copiado da view de Newton-Raphson, ajuste se necessário
                "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                "exp": sympy.exp, "ln": sympy.log, "log": sympy.log,
                "log10": lambda arg: sympy.log(arg, 10),
                "sqrt": sympy.sqrt, "abs": sympy.Abs, "fabs": sympy.Abs,
                "pi": sympy.pi, "e": sympy.E,
                "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
                "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
            }

            local_scope = allowed_functions.copy()
            local_scope['x'] = x_sym

            if not funcao_str:
                raise ValueError("A expressão da função não pode estar vazia.")

            func_sympy = sympy.sympify(funcao_str, locals=local_scope)
            
            if not isinstance(func_sympy, Expr):
                raise ValueError(f"A função '{funcao_str}' não foi interpretada como uma expressão matemática escalar válida.")

            if func_sympy.is_number:
                raise ValueError(f"A função fornecida é uma constante '{func_sympy}'. O método da bisseção busca raízes de funções variáveis.")

            func_callable = sympy.lambdify(x_sym, func_sympy, modules=['math'])

            # --- CÁLCULO DO MÉTODO DA BISSEÇÃO ---
            raiz, iteracoes, f_na_raiz, atingiu_max_iter, erro_calculado = metodo_bissecao(
                func_callable,
                val_a,
                val_b,
                erro,
                max_iter
            )
            
            # --- MONTA A MENSAGEM BASEADA NO CRITÉRIO DE PARADA ---
            if raiz is None:
                mensagem = "Erro: f(a) e f(b) devem ter sinais opostos."
            elif atingiu_max_iter:
                mensagem = "Máximo de iterações atingido."
            else:
                mensagem = "Convergiu pelo erro relativo."
            
            context['resultado'] = {
                'raiz': raiz, 
                'iteracoes': iteracoes, 
                'f_na_raiz': f_na_raiz,
                'erro_calculado': erro_calculado,
                'mensagem': mensagem,
            }

        # --- CAPTURA DE ERROS ---
        except (sympy.SympifyError, TypeError, NameError) as e:     # Captura erros do SymPy
            context['erro_sympy'] = f"Erro ao processar a função: '{e}'. Verifique a sintaxe."

        except ValueError as e:     # Captura erros de validação
            context['erro_sympy'] = str(e)

        except Exception as e:      # # Captura qualquer outro erro inesperado
            context['erro_sympy'] = f"Ocorreu um erro inesperado: {e}"

    # Renderiza a página
    # Se for GET, renderiza com os valores padrão
    # Se for POST, renderiza com os valores enviados e com o 'resultado' ou 'erro_sympy'
    return render(request, 'calculo/bissecao_calculator.html', context)


# --- Funções Auxiliares (Parser) de Gauss ---
def _parse_gauss_matriz(matriz_str: str) -> list:
    """ 
    Analisa o formato: '1 2 3; 4 5 6'
    """
    A = []
    if not matriz_str:
        return A
    
    matriz_str_limpa = matriz_str.strip().strip('[]')       # Remove colchetes, substitui vírgulas por espaços
    
    linhas = matriz_str_limpa.split(';')        # Divide as linhas pelo ';'
    
    for linha_str in linhas:
        linha_limpa = linha_str.strip()
        if not linha_limpa:
            continue
        
        valores = re.split(r'[,\s]+', linha_limpa)      # Substitui vírgulas por espaços e divide pelos espaços
        linha_float = [float(val) for val in valores if val.strip()]
        
        if linha_float:
            A.append(linha_float)
    return A

def _parse_gauss_vetor(vetor_str: str) -> list:
    """ 
    Analisa o formato: '3,4,2' OU '3 4 2'
    """
    if not vetor_str:
        return []
    
    vetor_limpo = vetor_str.strip().strip('[]')     # Remove colchetes, substitui vírgulas e ponto-e-vírgula por espaços
    valores = re.split(r'[;,\s]+', vetor_limpo)     # Divide por ';', ',' ou ' '
    
    b = [float(val) for val in valores if val.strip()]
    return b


# --- View da Calculadora de Gauss ---
def gauss_calculator_view(request):
    context = {
        'form_data': {      # Valores padrão
            'tamanho_matriz': '3x3',
            'matriz': '2 1 -1; -3 -1 2; -2 1 2',        # Formato da print
            'vetor': '8, -11, -3',      # Formato da print
        }
    }

    if request.method == 'POST':
        # --- PEGA OS DADOS USANDO OS 'name' --- 
        tamanho_matriz_str = request.POST.get('tamanho_matriz', '').strip()
        matriz_str = request.POST.get('matriz', '').strip()
        termos_str = request.POST.get('vetor', '').strip()
        
        context['form_data'] = {
            'tamanho_matriz': tamanho_matriz_str,
            'matriz': matriz_str,
            'vetor': termos_str,
        }
        
        metodo_alternativo = request.POST.get('metodo_alternativo') 

        try:
            # --- USO DAS FUNÇÕES AUXILIARES DE PARSE ---
            A = _parse_gauss_matriz(matriz_str)
            b = _parse_gauss_vetor(termos_str)

            # --- VALIDAÇÃO DOS DADOS ---
            if not A or not b:
                raise ValueError("Matriz A ou vetor b estão vazios.")

            if len(A) != len(b):
                raise ValueError(f"O número de linhas da matriz ({len(A)}) é diferente do número de termos no vetor b ({len(b)}).")
            
            if A:
                num_colunas = len(A[0])
                for i, linha in enumerate(A):
                    if len(linha) != num_colunas:
                        raise ValueError(f"A linha {i+1} da matriz tem {len(linha)} colunas, mas a primeira linha tem {num_colunas}.")

            # --- CHAMA A FUNÇÃO DE CÁLCULO ---
            resultado_dict = {}
            if metodo_alternativo == 'svd':
                resultado_dict = resolver_por_svd_web(A, b)
            elif metodo_alternativo == 'mq':
                 resultado_dict = resolver_por_minimos_quadrados_web(A, b)
            else:
                resultado_dict = gauss_somente_web(A, b)        # Tenta Gauss como padrão

            context['solucao'] = resultado_dict.get('solucao')
            context['mensagem'] = resultado_dict.get('mensagem')
            
            # --- SUGESTÕES PARA MÉTODOS ALTERNATIVOS ---
            if resultado_dict.get('status') == 'singular':
                context['sugerir_svd'] = True
            elif resultado_dict.get('status') == 'nao_quadrado':
                context['sugerir_mq'] = True

        except ValueError as e:     # Erro de formato
            context['erro_input'] = str(e)
        except Exception as e:
            context['erro_input'] = f"Ocorreu um erro inesperado: {e}"

    return render(request, 'calculo/gauss_calculator.html', context)