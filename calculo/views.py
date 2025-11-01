# calculo/views.py
from django.shortcuts import render
import sympy
from sympy.core.expr import Expr
from .bissecao_method import metodo_bissecao
from .newton_method import newton_raphson
from .gauss_method import gauss_somente

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

def gauss_calculator_view(request):
    context = {
        'form_data': { #valores padrão para o formulário na primeira carga
            'matriz_str': '2 1 -1\n-3 -1 2\n-2 1 2',
            'termos_str': '8\n-11\n-3',
        }
    }

    return render(request, 'calculo/gauss_calculator.html', context)