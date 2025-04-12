Entrega avaliativa de Inteligência Artificial, Sistemas de Informação - Unitins.
Falamos sobre o problema XOR, onde perceptron não podem diretamente fazer essa operação, por ser não linear. É preciso então dividir a operação em um numero mínimo de neurônios. 
Exploramos os parâmetros para que uma rede neural MLP possa fazer a operação XOR, emulando o comportamento de perceptrons montados manualmente, ela aprende sozinha os pesos ideais ao longo das épocas de treinamento, usando o algoritmo de backpropagation para minimizar o erro. 
A principal diferença entre a solução com perceptrons manuais e a MLP é a autonomia, já que enquanto a primeira exige que o programador defina a lógica da solução, a MLP descobre por meio de seu treinamento.

Para os parametros testes, foi feito os seguintes experimentos:

Cria a rede neural com uma camada oculta de 4 neurônios - parametros estaveis, confiaveis e com convergencia
nn = NeuralNetwork(input_size=2, hidden_sizes=[4], output_size=1, learning_rate=0.1, epochs=15000)

Parametros com camada menor que o necessário
nn = NeuralNetwork(input_size=2, hidden_sizes=[2], output_size=1, learning_rate=0.1, epochs=1000)

Camadas que tem convergencia mas pode causar erros
nn = NeuralNetwork(input_size=2, hidden_sizes=[3], output_size=1, learning_rate=0.5, epochs=5000)

Camadas insuficientes para convergencia, mesmo com muitos ciclos
nn = NeuralNetwork(input_size=2, hidden_sizes=[2], output_size=1, learning_rate=0.01, epochs=20000)

Camadas insuficientes para convergencia, mesmo com um alto aprendizado, permanece estagnado
nn = NeuralNetwork(input_size=2, hidden_sizes=[2], output_size=1, learning_rate=1.0, epochs=20000)
