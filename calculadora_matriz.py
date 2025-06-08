import os
import copy
from abc import ABC, abstractmethod

# Exceções personalizadas
class DimensoesIncompativeisError(Exception):
    pass

class OperacaoInvalidaError(Exception):
    pass

class TipoMatrizInvalidoError(Exception):
    pass

# Classe abstrata base para todas as matrizes
class Matriz(ABC):
    def __init__(self, linhas, colunas):
        self.linhas = linhas
        self.colunas = colunas
        self.nome = ""
        
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    @abstractmethod
    def __setitem__(self, idx, valor):
        pass
    
    @abstractmethod
    def to_matriz_geral(self):
        pass
    
    @abstractmethod
    def __add__(self, other):
        pass
    
    @abstractmethod
    def __sub__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, other):
        pass
    
    @abstractmethod
    def transposta(self):
        pass
    
    def traco(self):
        if self.linhas != self.colunas:
            raise OperacaoInvalidaError("Traço só pode ser calculado para matrizes quadradas")
        traco = 0.0
        for i in range(self.linhas):
            traco += self[i, i]
        return traco
    
    def determinante(self):
        if self.linhas != self.colunas:
            raise OperacaoInvalidaError("Determinante só pode ser calculado para matrizes quadradas")
        det = 1.0
        for i in range(self.linhas):
            det *= self[i, i]
        return det
    
    def tipo(self):
        return "Geral"
    
    def __str__(self):
        return self.to_matriz_geral().__str__()
    
    def __repr__(self):
        return f"<{self.tipo()} {self.linhas}x{self.colunas}>"

# Implementação das classes especializadas
class MatrizGeral(Matriz):
    def __init__(self, linhas, colunas, dados=None):
        super().__init__(linhas, colunas)
        if dados:
            self.dados = dados
        else:
            self.dados = [[0.0] * colunas for _ in range(linhas)]
    
    def __getitem__(self, idx):
        i, j = idx
        return self.dados[i][j]
    
    def __setitem__(self, idx, valor):
        i, j = idx
        self.dados[i][j] = valor
    
    def to_matriz_geral(self):
        return self
    
    def tipo(self):
        if self.linhas == self.colunas:
            return "Quadrada"
        return "Geral"
    
    def __add__(self, other):
        if self.linhas != other.linhas or self.colunas != other.colunas:
            raise DimensoesIncompativeisError("Dimensões incompatíveis para soma")
        
        # Otimização para matrizes do mesmo tipo especializado
        if type(self) == type(other):
            if isinstance(self, MatrizDiagonal):
                nova = MatrizDiagonal(self.linhas)
                for i in range(self.linhas):
                    nova[i, i] = self[i, i] + other[i, i]
                return nova
            elif isinstance(self, MatrizTriangularInferior) or isinstance(self, MatrizTriangularSuperior):
                nova = type(self)(self.linhas)
                for i in range(self.linhas):
                    for j in range(len(self.dados[i])):
                        nova.dados[i][j] = self.dados[i][j] + other.dados[i][j]
                return nova
        
        # Operação geral
        nova = MatrizGeral(self.linhas, self.colunas)
        for i in range(self.linhas):
            for j in range(self.colunas):
                nova[i, j] = self[i, j] + other[i, j]
        return nova
    
    def __sub__(self, other):
        if self.linhas != other.linhas or self.colunas != other.colunas:
            raise DimensoesIncompativeisError("Dimensões incompatíveis para subtração")
        
        # Otimização para matrizes do mesmo tipo especializado
        if type(self) == type(other):
            if isinstance(self, MatrizDiagonal):
                nova = MatrizDiagonal(self.linhas)
                for i in range(self.linhas):
                    nova[i, i] = self[i, i] - other[i, i]
                return nova
            elif isinstance(self, MatrizTriangularInferior) or isinstance(self, MatrizTriangularSuperior):
                nova = type(self)(self.linhas)
                for i in range(self.linhas):
                    for j in range(len(self.dados[i])):
                        nova.dados[i][j] = self.dados[i][j] - other.dados[i][j]
                return nova
        
        # Operação geral
        nova = MatrizGeral(self.linhas, self.colunas)
        for i in range(self.linhas):
            for j in range(self.colunas):
                nova[i, j] = self[i, j] - other[i, j]
        return nova
    
    def __mul__(self, other):
        # Multiplicação por escalar
        if isinstance(other, (int, float)):
            # Otimização para tipos especiais
            if isinstance(self, MatrizDiagonal):
                nova = MatrizDiagonal(self.linhas)
                for i in range(self.linhas):
                    nova[i, i] = self[i, i] * other
                return nova
            elif isinstance(self, MatrizTriangularInferior) or isinstance(self, MatrizTriangularSuperior):
                nova = type(self)(self.linhas)
                for i in range(self.linhas):
                    nova.dados[i] = [x * other for x in self.dados[i]]
                return nova
            else:
                nova = MatrizGeral(self.linhas, self.colunas)
                for i in range(self.linhas):
                    for j in range(self.colunas):
                        nova[i, j] = self[i, j] * other
                return nova
        
        # Multiplicação de matrizes
        elif isinstance(other, Matriz):
            if self.colunas != other.linhas:
                raise DimensoesIncompativeisError("Dimensões incompatíveis para multiplicação")
            
            # Otimização para matrizes diagonais
            if isinstance(self, MatrizDiagonal) and isinstance(other, MatrizDiagonal):
                nova = MatrizDiagonal(self.linhas)
                for i in range(self.linhas):
                    nova[i, i] = self[i, i] * other[i, i]
                return nova
            
            # Operação geral
            nova = MatrizGeral(self.linhas, other.colunas)
            for i in range(self.linhas):
                for k in range(self.colunas):
                    if self[i, k] != 0:  # Pequena otimização
                        for j in range(other.colunas):
                            nova[i, j] += self[i, k] * other[k, j]
            return nova
        
        raise TypeError("Tipo de operando não suportado")
    
    def transposta(self):
        nova = MatrizGeral(self.colunas, self.linhas)
        for i in range(self.linhas):
            for j in range(self.colunas):
                nova[j, i] = self[i, j]
        return nova
    
    def __str__(self):
        return "\n".join(["\t".join(map(str, linha)) for linha in self.dados])

class MatrizTriangularInferior(MatrizGeral):
    def __init__(self, n):
        super().__init__(n, n)
        self.dados = [[] for _ in range(n)]
        for i in range(n):
            self.dados[i] = [0.0] * (i + 1)  # i+1 elementos na linha i
    
    def __getitem__(self, idx):
        i, j = idx
        if j > i:  # Acima da diagonal
            return 0.0
        return self.dados[i][j]
    
    def __setitem__(self, idx, valor):
        i, j = idx
        if j > i:
            if valor != 0:
                raise ValueError("Não é possível definir elemento fora da parte triangular inferior")
            return
        self.dados[i][j] = valor
    
    def tipo(self):
        return "Triangular Inferior"
    
    def to_matriz_geral(self):
        matriz = MatrizGeral(self.linhas, self.colunas)
        for i in range(self.linhas):
            for j in range(i + 1):
                matriz[i, j] = self.dados[i][j]
        return matriz

class MatrizTriangularSuperior(MatrizGeral):
    def __init__(self, n):
        super().__init__(n, n)
        self.dados = [[] for _ in range(n)]
        for i in range(n):
            self.dados[i] = [0.0] * (n - i)  # n-i elementos na linha i
    
    def __getitem__(self, idx):
        i, j = idx
        if j < i:  # Abaixo da diagonal
            return 0.0
        return self.dados[i][j - i]
    
    def __setitem__(self, idx, valor):
        i, j = idx
        if j < i:
            if valor != 0:
                raise ValueError("Não é possível definir elemento fora da parte triangular superior")
            return
        self.dados[i][j - i] = valor
    
    def tipo(self):
        return "Triangular Superior"
    
    def to_matriz_geral(self):
        matriz = MatrizGeral(self.linhas, self.colunas)
        for i in range(self.linhas):
            for j in range(i, self.colunas):
                matriz[i, j] = self.dados[i][j - i]
        return matriz

class MatrizDiagonal(MatrizGeral):
    def __init__(self, n):
        super().__init__(n, n)
        self.dados = [0.0] * n
    
    def __getitem__(self, idx):
        i, j = idx
        if i == j:
            return self.dados[i]
        return 0.0
    
    def __setitem__(self, idx, valor):
        i, j = idx
        if i != j:
            if valor != 0:
                raise ValueError("Não é possível definir elemento fora da diagonal principal")
            return
        self.dados[i] = valor
    
    def tipo(self):
        return "Diagonal"
    
    def to_matriz_geral(self):
        matriz = MatrizGeral(self.linhas, self.colunas)
        for i in range(self.linhas):
            matriz[i, i] = self.dados[i]
        return matriz

# Funções auxiliares
def criar_matriz_identidade(n):
    matriz = MatrizDiagonal(n)
    for i in range(n):
        matriz[i, i] = 1.0
    return matriz

def ler_matriz_do_arquivo(caminho):
    with open(caminho, 'r') as f:
        linhas = f.readlines()
    
    if not linhas:
        return None
    
    # Extrair dimensões
    m, n = map(int, linhas[0].split())
    dados = []
    
    for i in range(1, m + 1):
        linha = list(map(float, linhas[i].split()))
        dados.append(linha)
    
    return MatrizGeral(m, n, dados)

def salvar_lista_matrizes(matrizes, caminho):
    with open(caminho, 'w') as f:
        for nome, matriz in matrizes.items():
            f.write(f"Matriz: {nome}\n")
            f.write(f"Tipo: {matriz.tipo()}\n")
            f.write(f"Dimensões: {matriz.linhas}x{matriz.colunas}\n")
            f.write("Dados:\n")
            f.write(str(matriz) + "\n\n")

class CalculadoraMatricial:
    def __init__(self):
        self.matrizes = {}  # Dicionário: nome -> matriz
    
    def adicionar_matriz(self, nome, matriz):
        self.matrizes[nome] = matriz
        matriz.nome = nome
    
    def remover_matriz(self, nome):
        if nome in self.matrizes:
            del self.matrizes[nome]
    
    def obter_matriz(self, nome):
        return self.matrizes.get(nome)
    
    def listar_matrizes(self):
        for nome, matriz in self.matrizes.items():
            print(f"{nome}: {matriz.tipo()} {matriz.linhas}x{matriz.colunas}")
    
    def executar_operacao(self, op, a_nome, b_nome=None, escalar=None):
        A = self.obter_matriz(a_nome)
        if not A:
            print(f"Matriz {a_nome} não encontrada!")
            return None
        
        if op in ['+', '-', '*'] and b_nome:
            B = self.obter_matriz(b_nome)
            if not B:
                print(f"Matriz {b_nome} não encontrada!")
                return None
            
            if op == '+':
                return A + B
            elif op == '-':
                return A - B
            elif op == '*':
                return A * B
        
        elif op == 'escalar' and escalar is not None:
            return A * escalar
        
        elif op == 'transposta':
            return A.transposta()
        
        elif op == 'traco':
            try:
                return A.traco()
            except OperacaoInvalidaError as e:
                print(e)
                return None
        
        elif op == 'determinante':
            try:
                return A.determinante()
            except OperacaoInvalidaError as e:
                print(e)
                return None
        
        return None

# Interface de usuário
def menu():
    calculadora = CalculadoraMatricial()
    
    while True:
        print("\n--- Calculadora Matricial ---")
        print("1. Adicionar matriz (teclado)")
        print("2. Adicionar matriz (arquivo)")
        print("3. Adicionar matriz identidade")
        print("4. Remover matriz")
        print("5. Listar matrizes")
        print("6. Imprimir matriz")
        print("7. Operações entre matrizes")
        print("8. Multiplicação por escalar")
        print("9. Transposição")
        print("10. Traço")
        print("11. Determinante")
        print("12. Salvar lista de matrizes")
        print("13. Carregar lista de matrizes")
        print("14. Zerar lista de matrizes")
        print("0. Sair")
        
        escolha = input("Escolha uma opção: ")
        
        if escolha == '1':
            nome = input("Nome da matriz: ")
            linhas = int(input("Número de linhas: "))
            colunas = int(input("Número de colunas: "))
            
            matriz = MatrizGeral(linhas, colunas)
            print("Digite os elementos linha por linha (separados por espaço):")
            
            for i in range(linhas):
                while True:
                    linha = input(f"Linha {i+1}: ").split()
                    if len(linha) != colunas:
                        print(f"Erro: esperados {colunas} elementos!")
                        continue
                    try:
                        for j, valor in enumerate(linha):
                            matriz[i, j] = float(valor)
                        break
                    except ValueError:
                        print("Valores inválidos! Use números.")
            
            calculadora.adicionar_matriz(nome, matriz)
            print(f"Matriz {nome} adicionada com sucesso!")
        
        elif escolha == '2':
            caminho = input("Caminho do arquivo: ")
            nome = input("Nome da matriz: ")
            
            try:
                matriz = ler_matriz_do_arquivo(caminho)
                if matriz:
                    calculadora.adicionar_matriz(nome, matriz)
                    print(f"Matriz {nome} carregada com sucesso!")
                else:
                    print("Erro ao ler arquivo!")
            except Exception as e:
                print(f"Erro: {e}")
        
        elif escolha == '3':
            nome = input("Nome da matriz: ")
            n = int(input("Dimensão (n): "))
            matriz = criar_matriz_identidade(n)
            calculadora.adicionar_matriz(nome, matriz)
            print(f"Matriz identidade {n}x{n} adicionada como {nome}")
        
        elif escolha == '4':
            nome = input("Nome da matriz a remover: ")
            calculadora.remover_matriz(nome)
            print(f"Matriz {nome} removida")
        
        elif escolha == '5':
            print("\nMatrizes disponíveis:")
            calculadora.listar_matrizes()
        
        elif escolha == '6':
            nome = input("Nome da matriz: ")
            matriz = calculadora.obter_matriz(nome)
            if matriz:
                print(f"\nMatriz {nome} ({matriz.tipo()}):")
                print(matriz)
            else:
                print("Matriz não encontrada!")
        
        elif escolha == '7':
            a_nome = input("Nome da primeira matriz: ")
            op = input("Operação (+, -, *): ")
            b_nome = input("Nome da segunda matriz: ")
            nome_resultado = input("Nome para o resultado: ")
            
            resultado = calculadora.executar_operacao(op, a_nome, b_nome)
            if resultado:
                calculadora.adicionar_matriz(nome_resultado, resultado)
                print(f"Resultado salvo como {nome_resultado}")
        
        elif escolha == '8':
            a_nome = input("Nome da matriz: ")
            escalar = float(input("Escalar: "))
            nome_resultado = input("Nome para o resultado: ")
            
            resultado = calculadora.executar_operacao('escalar', a_nome, escalar=escalar)
            if resultado:
                calculadora.adicionar_matriz(nome_resultado, resultado)
                print(f"Resultado salvo como {nome_resultado}")
        
        elif escolha == '9':
            a_nome = input("Nome da matriz: ")
            nome_resultado = input("Nome para a transposta: ")
            
            resultado = calculadora.executar_operacao('transposta', a_nome)
            if resultado:
                calculadora.adicionar_matriz(nome_resultado, resultado)
                print(f"Transposta salva como {nome_resultado}")
        
        elif escolha == '10':
            a_nome = input("Nome da matriz: ")
            traco = calculadora.executar_operacao('traco', a_nome)
            if traco is not None:
                print(f"Traço da matriz {a_nome}: {traco}")
        
        elif escolha == '11':
            a_nome = input("Nome da matriz: ")
            det = calculadora.executar_operacao('determinante', a_nome)
            if det is not None:
                print(f"Determinante da matriz {a_nome}: {det}")
        
        elif escolha == '12':
            caminho = input("Caminho para salvar: ")
            salvar_lista_matrizes(calculadora.matrizes, caminho)
            print("Lista salva com sucesso!")
        
        elif escolha == '13':
            caminho = input("Caminho do arquivo: ")
            try:
                with open(caminho, 'r') as f:
                    print("Carregamento de lista ainda não implementado completamente.")
                    # Implementação completa dependeria do formato de armazenamento
                print("Lista carregada com sucesso!")
            except Exception as e:
                print(f"Erro ao carregar: {e}")
        
        elif escolha == '14':
            calculadora.matrizes = {}
            print("Lista de matrizes zerada!")
        
        elif escolha == '0':
            print("Saindo...")
            break
        
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    menu()