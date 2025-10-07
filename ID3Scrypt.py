import pandas as pd
import numpy as np
from collections import Counter
import math

# Load, preprocess, and discretize the dataset

# path of the dataset
FILE_PATH = 'BBC_News_Train.csv'
df = pd.read_csv(r'C:\Users\sonic\OneDrive\Documents\IA Algorithms\P4_KNN_Class\BBC_News_Train.csv',sep=',')

print("Datos cargados. Filas:", len(df))
print("-" * 50)

# Calculating wordlength
df['WordCount'] = df['Text'].apply(lambda x: len(str(x).split()))

# Discretization, converting the wordcount in numerical level
bins = df['WordCount'].quantile([0.33, 0.66]).tolist()
bins = [df['WordCount'].min() - 1] + bins + [df['WordCount'].max() + 1]
labels = ['Corto', 'Medio', 'Largo']
df['Longitud_Nivel'] = pd.cut(df['WordCount'], bins=bins, labels=labels, include_lowest=True)


# Calculating the polarity based on a feeling (simulating the feeling)
df['Polaridad'] = np.select(
    [df['WordCount'] < df['WordCount'].mean() * 0.9,
     df['WordCount'] > df['WordCount'].mean() * 1.1],
    ['Negativa', 'Positiva'],
    default='Neutra'
)

# Select only categories
df_id3 = df[['Longitud_Nivel', 'Polaridad', 'Category']]
df_id3 = df_id3.rename(columns={'Category': 'Clase'})

print("Características Discretizadas (primeras 5 filas):")
print(df_id3.head())
print("-" * 50)

# Entropy and information gain 

def calculate_entropy(data, target_attribute):
    """Calculates the entropy of a data group (column 'Class')."""
    
    # Count the frequency of each class
    class_counts = data[target_attribute].value_counts().to_dict()
    total_samples = len(data)
    
    entropy = 0
    
    # Iterate over each sample to earn entropy E = -Σ(p_i * log2(p_i))
    for count in class_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)
            
    return entropy

def calculate_information_gain(data, feature_attribute, target_attribute="Clase"):
    """Calculates the information gain of the data group"""
    
    # Total entropy from parent
    total_entropy = calculate_entropy(data, target_attribute)
    total_samples = len(data)
    
    # Mean entropy from childs
    weighted_entropy = 0
    
    # Unique values (ej: 'Corto', 'Medio', 'Largo')
    for value in data[feature_attribute].unique():
        
        # Filter the group
        subset = data[data[feature_attribute] == value]
        subset_entropy = calculate_entropy(subset, target_attribute)
        
        # Mean of the filter
        weight = len(subset) / total_samples
        weighted_entropy += weight * subset_entropy
        
    # Information gain Gain(A) = Entropy(S) - Σ((|Sv|/|S|) * Entropy(Sv))
    information_gain = total_entropy - weighted_entropy
    
    return information_gain

def find_best_split_attribute(data, attributes):
    """Finds the best information gain."""
    
    if not attributes:
        return None
        
    gains = {}
    for attribute in attributes:
        gain = calculate_information_gain(data, attribute)
        gains[attribute] = gain
        
    # Best gain for optimize
    best_attribute = max(gains, key=gains.get)
    return best_attribute

# Struct for the tree node
class Node:
    def __init__(self, attribute=None, value=None, children=None, is_leaf=False, classification=None):
        self.attribute = attribute    # Dividing attribute
        self.value = value            # Value of the node(in case of child)
        self.children = children if children is not None else {} # childs (sub-trees)
        self.is_leaf = is_leaf        
        self.classification = classification
        
# Recursive tree development (ID3)

def build_id3_tree(data, attributes, target_attribute="Clase"):
    """Builds ID3 recursive tree."""
    
    # Obtains all the classes 
    classes = data[target_attribute].unique()
    
    # Case 1: raw
    if len(classes) == 1:
        return Node(is_leaf=True, classification=classes[0])
    
    # Case2 : No attributes left
    if not attributes or data.empty:
        # Devolver la clasificación mayoritaria (voto)
        majority_class = data[target_attribute].mode()[0]
        return Node(is_leaf=True, classification=majority_class)
    
    # Case 3: Find best divider
    best_attribute = find_best_split_attribute(data, attributes)
    
    # Create root node
    root = Node(attribute=best_attribute)
    
    # Delete selected attribute
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Divide and create new childs
    for value in data[best_attribute].unique():
        
        #Create Sv Subgroup
        subset = data[data[best_attribute] == value].drop(columns=[best_attribute])
        
        # Recursive call for the tree
        child_node = build_id3_tree(subset, remaining_attributes, target_attribute)
        
        # Assign the value for the characteristic
        child_node.value = value 
        
        # Connect child to root
        root.children[value] = child_node
        
    return root

# Start building the tree
attributes_list = list(df_id3.columns.drop('Clase'))
decision_tree = build_id3_tree(df_id3.copy(), attributes_list)

#Functions of entropy and information gain

def calculate_entropy(data, target_attribute):
    """Calcula la entropía de un conjunto de datos (columna 'Clase')."""
    
    # Count frequency
    class_counts = data[target_attribute].value_counts().to_dict()
    total_samples = len(data)
    
    entropy = 0
    
    # Iterate on each for calculate entropy. E = -Σ(p_i * log2(p_i))
    for count in class_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)
            
    return entropy

def calculate_information_gain(data, feature_attribute, target_attribute="Clase"):
    """Calculates information gain"""
    
    # Total entropy
    total_entropy = calculate_entropy(data, target_attribute)
    total_samples = len(data)
    
    # Childs
    weighted_entropy = 0
    
    # Valores únicos de la característica (ej: 'Corto', 'Medio', 'Largo')
    for value in data[feature_attribute].unique():
        
        # Filter the subgroup
        subset = data[data[feature_attribute] == value]
        subset_entropy = calculate_entropy(subset, target_attribute)
        
        # Subgroup mean
        weight = len(subset) / total_samples
        weighted_entropy += weight * subset_entropy
        
    # 3. Information gain: Gain(A) = Entropy(S) - Σ((|Sv|/|S|) * Entropy(Sv))
    information_gain = total_entropy - weighted_entropy
    
    return information_gain

def find_best_split_attribute(data, attributes):
    """Fins the best information gain"""
    
    if not attributes:
        return None
        
    gains = {}
    for attribute in attributes:
        gain = calculate_information_gain(data, attribute)
        gains[attribute] = gain
        
    # Best attribute for information gain
    best_attribute = max(gains, key=gains.get)
    return best_attribute

# Node struct for the tree
class Node:
    def __init__(self, attribute=None, value=None, children=None, is_leaf=False, classification=None):
        self.attribute = attribute    # Característica que se está dividiendo
        self.value = value            # Valor de la característica para este nodo (si es hijo)
        self.children = children if children is not None else {} # Hijos (sub-árboles)
        self.is_leaf = is_leaf        # Es un nodo hoja?
        self.classification = classification # Clasificación si es nodo hoja
        
# Building of the ID3 tree

def build_id3_tree(data, attributes, target_attribute="Clase"):
    """Builds a recursive ID3 tree"""
    
    # Obtain all the classes from data group
    classes = data[target_attribute].unique()
    
    # Case 1: Pure element
    if len(classes) == 1:
        return Node(is_leaf=True, classification=classes[0])
    
    # Case 2: No remain attributes
    if not attributes or data.empty:

        majority_class = data[target_attribute].mode()[0]
        return Node(is_leaf=True, classification=majority_class)
    
    # Step 3: Find best attribute
    best_attribute = find_best_split_attribute(data, attributes)
    
    # Create actual node root
    root = Node(attribute=best_attribute)
    
    # Delete the selected attribute
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Case 4: Divide and create childs
    for value in data[best_attribute].unique():
        
        # Create sv subgroup
        subset = data[data[best_attribute] == value].drop(columns=[best_attribute])
        
        # Recursive call
        child_node = build_id3_tree(subset, remaining_attributes, target_attribute)
        
        # Assing the value to the characteristic to child node for the path
        child_node.value = value 
        
        # Connect child node to root
        root.children[value] = child_node
        
    return root

# Start building the tree
attributes_list = list(df_id3.columns.drop('Clase'))
decision_tree = build_id3_tree(df_id3.copy(), attributes_list)

#Prediction and visualization of the tree

def predict_id3(node, sample):
    """Clasifies a new sample of the ID3 tree."""
    
    if node.is_leaf:
        return node.classification
    
    # Obtain the value of the sample 
    attribute_value = sample[node.attribute]
    
    if attribute_value in node.children:
        # move to next node
        return predict_id3(node.children[attribute_value], sample)
    else:
        # Treat values never seen (assuming leaf node information)
        return "Clase Desconocida (Valor no visto)"

def print_tree(node, depth=0):
    """Imprime la estructura del árbol de decisión."""
    
    indent = "  | " * depth
    
    if node.is_leaf:
        print(f"{indent}-> CLASIFICACIÓN: {node.classification}")
        return
        
    print(f"{indent}SPLIT en {node.attribute} (Profundidad {depth}):")
    
    for value, child in node.children.items():
        print(f"{indent}  Valor '{value}':")
        print_tree(child, depth + 1)
        
# Execution and test

print("Starting ID3 construction (Based in 2 discrete characteristics)")
print("-" * 75)
print_tree(decision_tree)
print("-" * 75)

# Create a sample test with the same fields
sample_data = {
    'Longitud_Nivel': 'Largo',
    'Polaridad': 'Positiva'
}
sample_df = pd.Series(sample_data)

prediction = predict_id3(decision_tree, sample_df)
print(f"Predicción para Longitud='{sample_data['Longitud_Nivel']}', Polaridad='{sample_data['Polaridad']}' -> {prediction}")