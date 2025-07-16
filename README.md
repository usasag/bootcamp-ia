# Classificação de Cães e Gatos com Transfer Learning

Este projeto implementa um classificador de imagens para distinguir entre cães e gatos utilizando a técnica de Transfer Learning com a arquitetura VGG19 pré-treinada na base de dados ImageNet.

## Visão Geral

O objetivo deste projeto é criar um modelo de aprendizado de máquina capaz de classificar imagens em duas categorias: cães ou gatos. Para isso, utilizamos Transfer Learning com a arquitetura VGG19, que foi pré-treinada na base de dados ImageNet.

## Requisitos

Para executar este projeto, você precisará ter instalado:

- Python 3.8 ou superior
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV
- Requests
- Jupyter Notebook (para visualizar o notebook original)

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/usasag/bootcamp-ia.git
   cd bootcamp-ia
   ```

2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows use: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Estrutura do Projeto

```
bootcamp-ia/
├── DIO_Projeto_Transfer_Learning.ipynb  # Notebook principal
├── dogs_cats.keras                      # Modelo treinado
├── dogs_cats.weights.h5                 # Pesos do modelo
├── cats_vs_dogs_dataset/                # Conjunto de dados
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
└── README.md
```

## Como Usar

### Treinamento do Modelo

1. Execute o notebook `DIO_Projeto_Transfer_Learning.ipynb`:
   ```bash
   jupyter notebook DIO_Projeto_Transfer_Learning.ipynb
   ```

2. Siga as células do notebook para:
   - Baixar e preparar o conjunto de dados
   - Configurar o modelo VGG19 com Transfer Learning
   - Treinar o modelo
   - Salvar o modelo treinado

### Realizando Previsões

Para classificar uma imagem como cão ou gato, utilize a função de predição fornecida no notebook ou crie um script Python:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo
model = load_model('dogs_cats.keras')

def classificar_imagem(caminho_imagem):
    # Carregar e pré-processar a imagem
    img = image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Fazer a previsão
    predicao = model.predict(img_array)
    
    # Interpretar o resultado
    if predicao[0][0] > 0.5:
        return f"Cachorro com {predicao[0][0]*100:.2f}% de confiança"
    else:
        return f"Gato com {(1-predicao[0][0])*100:.2f}% de confiança"

# Exemplo de uso
print(classificar_imagem('caminho/para/sua/imagem.jpg'))
```

## Conjunto de Dados

O conjunto de dados utilizado é o [Dogs vs Cats](https://www.microsoft.com/en-us/download/details.aspx?id=54765) da Microsoft, que contém 25.000 imagens de cães e gatos (12.500 de cada classe).

## Resultados

O modelo atinge uma acurácia de aproximadamente 90-95% no conjunto de validação após o treinamento.

## Melhorias Futuras

- Aumentar o tamanho do conjunto de dados com técnicas de data augmentation
- Experimentar outras arquiteturas de redes neurais (ResNet, Inception, etc.)
- Implementar uma interface web para facilitar o uso do modelo
- Adicionar suporte para mais classes de animais

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para abrir uma issue ou entrar em contato.

---

Desenvolvido como parte do Bootcamp de IA da DIO.
