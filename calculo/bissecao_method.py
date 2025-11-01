# calculo/bissecao_method.py

def metodo_bissecao(func, a, b, erro=1e-7, i_max=100, prev_m=None, iter_count=0):
    """
    Encontra a raiz de uma função usando o método da Bisseção (versão recursiva).
    Retorna: (raiz, iteracoes, f(raiz), atingiu_max_iter, erro_calculado)
    """
    ## Primeira chamada: verifica se f(a) e f(b) têm sinais opostos
    if prev_m is None and iter_count == 0: #caso o ponto medio (prev_m) seja None e o contador de iterações seja 0
        y_a_inicial = func(a) # y_a_inicial recebe a função avaliada em a
        y_b_inicial = func(b) # y_b_inicial recebe a função avaliada em b
        if y_a_inicial * y_b_inicial >= 0: #verifica se os sinais são iguais
            return None, 0, None, False, None # Retorna None se os sinais não forem opostos
        
    # A função "func" é chamada para avaliar os pontos a, b e m
    y_a = func(a) # y_a recebe a função avaliada em a
    y_b = func(b) # y_b recebe a função avaliada em b

    m = (a + b) / 2.0 # m é o ponto médio do intervalo [a, b]/2
    y_m = func(m) # y_m recebe a função avaliada em m

    iter_count += 1 # Incrementa o contador de iterações apos definir m

    ## Calcula o erro relativo
    erro_calculado = abs((m - prev_m)/m) if prev_m is not None else None # calcula o erro relativo se prev_m não for None. "abs" serve como modulo
    
    ## Critério de parada: erro relativo
    if prev_m is not None and erro_calculado < erro: #verifica se o erro calculado é menor que o erro permitido
        return m, iter_count, y_m, False, erro_calculado 
    
    ## Critério de parada: f(m) = 0
    if y_m == 0: 
        return m, iter_count, y_m, False, 0.0
    
    ## Critério de parada: iterações esgotadas
    if iter_count >= i_max:
        return m, iter_count, y_m, True, erro_calculado
    
    ## Escolhe o novo intervalo
    if (y_a * y_m < 0): #verifica se a multiplicação dos sinais é negativa
        b = m
    elif (y_b * y_m < 0): #verifica se a multiplicação dos sinais é negativa
        a = m
    else:
        return m, iter_count, y_m, False, erro_calculado
    
    return metodo_bissecao(func, a, b, erro, i_max, m, iter_count)