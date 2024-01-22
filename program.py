import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# Chargement du CSV
csv_file = 'file.csv'
data = pd.read_csv(csv_file)

# Initialisation de votre suite de valeurs
a = 2
x0 = 0.1
values = data['Value'][:500]

# Génération de la suite
T = 1
suite = data['Value'][:500]

# Architecture du reseau par Takens
def teta(vector):
    teta_matrix = np.zeros((500, 500))
    for k in range(500):
        for x in range(500):
            teta_matrix[k, x] = vector[k] * vector[x]
    return teta_matrix

def all_covariance(vector):
    cov_matrix = teta(vector)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    print(eigenvalues);
    reversed_eigenvalues = sorted(eigenvalues, reverse=True)
    E = []
    for i in range(len(reversed_eigenvalues)-1):
        sqrtEigVal = math.sqrt(abs(reversed_eigenvalues[i+1]))
        E.append(sqrtEigVal)
    print(E)

    plt.plot(E, marker='*', linestyle='')
    plt.xlabel('Indices')
    plt.ylabel('Valeurs')
    plt.title('Graphique de Valeurs Triées El ')
    plt.grid(True)
    plt.show()
    return eigenvalues;
all_covariance(suite);

# Fonction d'activation
def g(sum):
    val = 1 / (1 + math.exp(-sum))
    return round(val, 8)

# Dérivée de la fonction d'activation
def gprime(sum):
    return round((g(sum) * (1 - g(sum))), 8)

# Fonction h pour calculer les activations
def h(i, m, oldWeight, initialValues, unites):
    value = 0
    for j in range(unites[m - 1]):
        value += oldWeight[m - 1][i][j] * initialValues[m - 1][j]
    return value

# Calcul du delta des poids
def deltaWeight(i, j, m, deltas, initialValues, N):
    dWeight = N * deltas[m - 1][i] * initialValues[m - 1][j]
    return round(dWeight, 8)

# Descente de gradient
def gradientDescent(unites, initialValues, oldWeight, desired_output):
    M = 3
    N = 0.2
    newWeight = [
        [
            [0, 0]
        ],
        [
            [0]
        ]
    ]
    xsortie = 0;
    deltas = [
        [0],
        [0]
    ]
    # Calcul des valeurs de la couche cachee et de la couche de sortie
    try:
        m=1
        for i in range(unites[m]):
            value = h(i, m, oldWeight, initialValues, unites)
            initialValues[m][i] = round(g(value), 8)
        m=2
        for i in range(unites[m]):
            value = h(i, m, oldWeight, initialValues, unites)
            initialValues[m][i] = round(value, 8)
        print("Vo :",initialValues)
        xsortie = initialValues[2][0]
        print("xsortie: ",xsortie);
    except Exception as error:
        print("Error at step three:")
        print(error)

    # Calcul de deltas pour les couches de sortie
    try:
        value = 0
        for i in range(unites[M - 1]):
            value = gprime(h(i, M - 1, oldWeight, initialValues, unites)) * (desired_output - initialValues[2][i])
            deltas[M - 2][i] = round(value, 8)
    except Exception as error:
        print("Error at step four:")
        print(error)
    
    print("deltas :",deltas)
    # Calcul des deltas de la couche cachee
    try:
        for i in range(unites[M - 2]):
            value = 0
            for j in range(unites[M - 1]):
                value += oldWeight[M - 2][j][i] * deltas[M - 2][j]
            adelta = gprime(h(i, M - 2, oldWeight, initialValues, unites)) * value
            deltas[M - 3][i] = round(adelta, 8)
    except Exception as error:
        print("Error at step five:")
        print(error)
    print(deltas)

    # Calcul des nouveaux poids
    try:
        for m in range(M - 1, 0, -1):
            for i in range(unites[m]):
                for j in range(unites[m - 1]):
                    deltaWeightValue = deltaWeight(i, j, m, deltas, initialValues, N)
                    value = deltaWeightValue + oldWeight[m - 1][i][j]
                    newWeight[m - 1][i][j] = round(value, 8)
        print('Nouveaux poids : ')
        print(newWeight)
        return {"xsortie" : xsortie, "newWeight": newWeight}
    except Exception as error:
        print("Error at step six:")
        print(error)

# Calcul de la variance
def variance(suite):
    moy = sum(suite) / len(suite)
    somme = 0
    for i in range(len(suite)):
        somme += pow((suite[i] - moy), 2)
    var = somme / len(suite)
    return var

# Calcul du NMSE
def nmse(suite, xsorties, prototypes):
    somme = 0
    for i in range(prototypes-2):
        somme += pow((suite[i+2] - xsorties[i]), 2)
    error = somme / (len(suite) * variance(suite))
    return error


# Fonction pour l'apprentissage sur plusieurs périodes
def periods(generatedValues, epoch, prototypes) :
    unites = [2,1,1];
    oldWeight = [
        [
            [0.9, 0.5],
        ],
        [
            [0.3]
        ]
    ]
    result = {};
    errors = [];
    error = 0;
    min_error = float('inf')
    min_error_count = 0
    best_weights = []
    for i in range(epoch) :
        xsorties = []
        for j in range (prototypes-2) :
            initialValues = [
                [generatedValues[j], generatedValues[j+1]],
                [0],
                [0]
            ]
            desired_output = generatedValues[j+2]
            result = gradientDescent(unites, initialValues, oldWeight, desired_output);
            oldWeight = result["newWeight"];
            xsorties.append(result["xsortie"])
        error = nmse(generatedValues, xsorties, prototypes)
        errors.append(error)

        if error < min_error:
            min_error = error
            min_error_count = 0
            best_weights = oldWeight.copy()
        else:
            min_error_count += 1
        if min_error_count >= 3:
            break

    print("Minimum des erreurs:", f"{min_error:.8f}")
    print("Erreurs:", errors) 
    plt.plot(errors, marker='o', linestyle='-')
    plt.xlabel('Indices')
    plt.ylabel('Valeurs')
    plt.title('NMSE')
    plt.grid(True)
    plt.show()
    return best_weights

epoch = 20
prototypes = 300
newWeight = periods(suite, epoch, prototypes)

# Prediction

values = suite[:11];
unites = [2, 1, 1]

def hh(i, m, weight, prototypes, unites):
    value = 0
    for j in range(unites[m - 1]):
        value += weight[m - 1][i][j] * prototypes[j]
    return value

def g(sum):
    val = 1 / (1 + math.exp(-sum))
    print(val);
    return round(val, 8)

def oneStepPred(prototypes, weight, unites):
    predictedValue = 0
    hiddenValues = [0] * unites[1]
    value = 0
    temp = 0
    for i in range(unites[1]):
        temp = g(hh(i, 1, weight, prototypes, unites))
        hiddenValues[i] = temp
    for j in range(unites[2]):
        value = hh(j, 2, weight, hiddenValues, unites)
    predictedValue = round(value, 8)
    return predictedValue

# Prediction a un pas en avant 
def prediction(values, weight, unites):
    predictedValues = [0] * (len(values) - 1)
    for i in range(len(values) - 1):
        prototypes = [values[i], values[i + 1]]
        predictedValue = oneStepPred(prototypes, weight, unites)
        predictedValues[i] = predictedValue
        print(predictedValue);
    return predictedValues

onestep = prediction(values,newWeight,unites);
onestep_indice = list(range(10))
waited_values = [];
for i in range(2, 12):
    waited_values.append(suite[i])
plt.plot(onestep_indice, onestep, marker='o', linestyle='-', label='onestep')
plt.plot(onestep_indice, waited_values, marker='x', linestyle='-', label='generated_values')
plt.xlabel('Indices')
plt.ylabel('Valeurs')
plt.title('Predictions a un pas en avant de 10 valeurs')
plt.grid(True)
plt.show()

# Prediction a plusieurs pas en avant 

def fewStepsPred(pas, values, weight):
    prototypes = [values[0], values[1]]
    predictedValues = []  # Commence a x2 et se termine a x12 si 10 predictions
    for k in range(pas):
        temp = prototypes
        result = oneStepPred(temp, weight, unites)
        temp[1] = temp[0]
        temp[0] = result
        predictedValues.append(result)
    return predictedValues

# Prediction a 3 pas en avant 
pas = 3
steps = fewStepsPred(pas, values, newWeight);
step_values = [];
for i in range(2, 5):
    step_values.append(suite[i])
steps_indice = list(range(pas));
plt.plot(steps_indice, steps, marker='o', linestyle='-', label='onestep')
plt.plot(steps_indice, step_values, marker='x', linestyle='-', label='generated_values')
plt.xlabel('Indices')
plt.ylabel('Valeurs')
plt.title('Predictions a 3 pas en avant de 02 valeurs: ')
plt.grid(True)
plt.show()

# Prediction a 10 pas en avant 
pas = 10
steps = fewStepsPred(pas, values, newWeight);

step_values = [];
for i in range(2, 12):
    step_values.append(suite[i])
steps_indice = list(range(pas));
plt.plot(steps_indice, steps, marker='o', linestyle='-', label='onestep')
plt.plot(steps_indice, step_values, marker='x', linestyle='-', label='generated_values')
plt.xlabel('Indices')
plt.ylabel('Valeurs')
plt.title('Predictions a 10 pas en avant de 02 valeurs: ')
plt.grid(True)
plt.show()

# Prediction a 20 pas en avant 
pas = 20
steps = fewStepsPred(pas, values, newWeight);

step_values = [];
for i in range(2, 22):
    step_values.append(suite[i])
steps_indice = list(range(pas));
plt.plot(steps_indice, steps, marker='o', linestyle='-', label='onestep')
plt.plot(steps_indice, step_values, marker='x', linestyle='-', label='generated_values')
plt.xlabel('Indices')
plt.ylabel('Valeurs')
plt.title('Predictions a 20 pas en avant de 02 valeurs: ')
plt.grid(True)
plt.show()

