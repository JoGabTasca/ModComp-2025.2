from django.shortcuts import render
import sympy
from sympy.core.expr import Expr
from .bissecao_method import metodo_bissecao
from .newton_method import newton_raphson
from .gauss_method import gauss_somente_web, resolver_por_svd_web, resolver_por_minimos_quadrados_web
import numpy as np 
import re 



def home_calculo_view(request):
    """
    View para a página inicial do app 'calculo', onde o usuário escolhe o método.
    """
    return render(request, 'calculo/home_calculo.html')


def newton_calculator_view(request):
    context = {
        'form_data': { #valores padrão para o formulário na primeira carga
            'funcao_str': 'x**2 - 4',
            'x0_str': '1.0',
            'erro_str': '1e-7',
            'max_iter_str': '100',
        }
    }

    derivada_calculada_str = ""
    f_na_raiz = None

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

        context['form_data'] = { #atualiza com os dados enviados
            'funcao_str': funcao_str,
            'x0_str': x0_str,
            'erro_str': erro_str,
            'max_iter_str': max_iter_str,
        }

        try:
            #validação e conversão dos inputs numéricos
            if not x0_str or not erro_str or not max_iter_str:
                raise ValueError("Todos os campos numéricos (x0, tolerância, máx. iterações) são obrigatórios.")
            
            x0 = float(x0_str.replace(',', '.'))
            erro = float(erro_str.replace(',', '.'))
            max_iter = int(max_iter_str)

            if erro <= 0:
                raise ValueError("A tolerância deve ser um valor positivo.")
            if max_iter <= 0:
                raise ValueError("O número máximo de iterações deve ser positivo.")

        except ValueError as e:
            context['erro_input'] = f"Erro nos valores numéricos: {e} Verifique se usou ponto '.' como separador decimal ou se os valores são válidos."
            return render(request, 'calculo/newton_calculator.html', context)

        try:
            print("DEBUG DJANGO VIEW: Entrando no bloco try do SymPy...") #DEBUG
            x_sym = sympy.symbols('x')
            
            allowed_functions = {
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
            print(f"DEBUG DJANGO VIEW: Antes do sympify, funcao_str: '{funcao_str}'") #DEBUG

            func_sympy = sympy.sympify(funcao_str, locals=local_scope)
            print(f"DEBUG DJANGO VIEW: Após sympify, func_sympy: {func_sympy}") #DEBUG
            
            if not isinstance(func_sympy, Expr):
                raise ValueError(f"A função '{funcao_str}' não foi interpretada como uma expressão matemática escalar válida. Verifique a sintaxe.")

            if func_sympy.is_number:
                if sympy.Eq(func_sympy, 0):
                    raise ValueError("A função fornecida é '0'. Não é possível aplicar Newton-Raphson.")
                else:
                    raise ValueError(f"A função fornecida é uma constante '{func_sympy}'. Não há raízes (a menos que a constante seja 0).")

            #a verificação if x_sym not in func_sympy.free_symbols: foi removida por enquanto
            #pois a lógica de derivada zero e is_number deve cobrir os casos problemáticos.

            derivada_sympy = sympy.diff(func_sympy, x_sym)
            print(f"DEBUG DJANGO VIEW: Após diff, derivada_sympy: {derivada_sympy}") #DEBUG
            derivada_calculada_str = str(derivada_sympy)

            func_callable = sympy.lambdify(x_sym, func_sympy, modules=['math'])
            print(f"DEBUG DJANGO VIEW: func_callable criada: {func_callable}") #DEBUG
            derivada_callable = sympy.lambdify(x_sym, derivada_sympy, modules=['math'])
            print(f"DEBUG DJANGO VIEW: derivada_callable criada: {derivada_callable}") #DEBUG

            print(f"DEBUG DJANGO VIEW: Antes de chamar newton_raphson com x0={x0}, tol={erro}") #DEBUG
            raiz, iteracoes, f_na_raiz, atingiu_max_iter, erro_calculado = newton_raphson(
                func_callable,
                derivada_callable,
                x0,
                erro,
                max_iter
            )
            print(f"DEBUG DJANGO VIEW: Após newton_raphson: raiz={raiz}, iter={iteracoes}") #DEBUG
            
            # Monta a mensagem baseada no critério de parada
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

        except (sympy.SympifyError, TypeError, NameError) as e:
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (SympifyError, TypeError, NameError): {type(e).__name__} - {e}") #DEBUG
            context['erro_sympy'] = f"Erro ao processar a função: '{e}'. Verifique a sintaxe. Use 'x' como variável e funções como sin(x), exp(x), log(x), etc."
        except ValueError as e:
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (ValueError): {e}") #DEBUG
            context['erro_sympy'] = str(e)
        except Exception as e:
            print(f"DEBUG DJANGO VIEW: ERRO CAPTURADO (Outra Exceção): {type(e).__name__} - {e}") #DEBUG
            context['erro_sympy'] = f"Ocorreu um erro inesperado no processamento da função: {e}"

    return render(request, 'calculo/newton_calculator.html', context)

def bissecao_calculator_view(request):
    context = {
        'form_data': { #valores padrão para o formulário na primeira carga
            'funcao_str': 'x**3 - x - 2',
            'a_str': '1.0',
            'b_str': '2.0',
            'erro_str': '1e-5',
            'max_iter_str': '100',
        }
    }
    f_na_raiz = None #para armazenar o valor de f(raiz_encontrada)

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

        try:
            if not a_str or not b_str or not erro_str or not max_iter_str:
                raise ValueError("Todos os campos numéricos (a, b, tolerância, máx. iterações) são obrigatórios.")
            
            val_a = float(a_str.replace(',', '.'))
            val_b = float(b_str.replace(',', '.'))
            erro = float(erro_str.replace(',', '.'))
            max_iter = int(max_iter_str)

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
            allowed_functions = { #copiado da view de newton, ajuste se necessário
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

            #chama o método da bisseção
            raiz, iteracoes, f_na_raiz, atingiu_max_iter, erro_calculado = metodo_bissecao(
                func_callable,
                val_a,
                val_b,
                erro,
                max_iter
            )
            
            # Monta a mensagem baseada no critério de parada
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

        except (sympy.SympifyError, TypeError, NameError) as e:
            context['erro_sympy'] = f"Erro ao processar a função: '{e}'. Verifique a sintaxe."
        except ValueError as e:
            context['erro_sympy'] = str(e)
        except Exception as e:
            context['erro_sympy'] = f"Ocorreu um erro inesperado: {e}"
            
    return render(request, 'calculo/bissecao_calculator.html', context)


# Gauss

def _parse_gauss_matriz(matriz_str: str) -> list:
    """ Analisa o formato: '1 2 3; 4 5 6' """
    A = []
    if not matriz_str:
        return A
    
    # Remove colchetes, substitui vírgulas por espaços
    matriz_str_limpa = matriz_str.strip().strip('[]')
    
    # Divide as linhas pelo ';'
    linhas = matriz_str_limpa.split(';')
    
    for linha_str in linhas:
        linha_limpa = linha_str.strip()
        if not linha_limpa:
            continue
        
        # Substitui vírgulas por espaços e divide pelos espaços
        valores = re.split(r'[,\s]+', linha_limpa)
        linha_float = [float(val) for val in valores if val.strip()]
        
        if linha_float:
            A.append(linha_float)
    return A

def _parse_gauss_vetor(vetor_str: str) -> list:
    """ Analisa o formato: '3,4,2' OU '3 4 2' """
    if not vetor_str:
        return []
    
    # Remove colchetes, substitui vírgulas E ponto-e-vírgula por espaços
    vetor_limpo = vetor_str.strip().strip('[]')
    valores = re.split(r'[;,\s]+', vetor_limpo) # Divide por ';', ',' ou ' '
    
    b = [float(val) for val in valores if val.strip()]
    return b

def gauss_calculator_view(request):
    context = {
        'form_data': { 
            'tamanho_matriz': '3x3',
            'matriz': '2 1 -1; -3 -1 2; -2 1 2', # Formato da print
            'vetor': '8, -11, -3',             # Formato da print
        }
    }

    if request.method == 'POST':
        # 1. Pega os dados usando os 'name' 
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
            
            A = _parse_gauss_matriz(matriz_str)
            b = _parse_gauss_vetor(termos_str)

            if not A or not b:
                raise ValueError("Matriz A ou vetor b estão vazios.")

            if len(A) != len(b):
                raise ValueError(f"O número de linhas da matriz ({len(A)}) é diferente do número de termos no vetor b ({len(b)}).")
            
            if A:
                num_colunas = len(A[0])
                for i, linha in enumerate(A):
                    if len(linha) != num_colunas:
                        raise ValueError(f"A linha {i+1} da matriz tem {len(linha)} colunas, mas a primeira linha tem {num_colunas}.")

            # 3. CHAMA A FUNÇÃO DE CÁLCULO
            resultado_dict = {}
            if metodo_alternativo == 'svd':
                resultado_dict = resolver_por_svd_web(A, b)
            elif metodo_alternativo == 'mq':
                 resultado_dict = resolver_por_minimos_quadrados_web(A, b)
            else:
                resultado_dict = gauss_somente_web(A, b) # Tenta Gauss como padrão

    
            context['solucao'] = resultado_dict.get('solucao')
            context['mensagem'] = resultado_dict.get('mensagem')
            
            # Sugestões para métodos alternativos
            if resultado_dict.get('status') == 'singular':
                context['sugerir_svd'] = True
            elif resultado_dict.get('status') == 'nao_quadrado':
                context['sugerir_mq'] = True

        except ValueError as e:
            context['erro_input'] = str(e) # Erro de formato
        except Exception as e:
            context['erro_input'] = f"Ocorreu um erro inesperado: {e}"

    return render(request, 'calculo/gauss_calculator.html', context)
