import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# Load training features
X_train = pd.read_csv("traindata.txt", header=None, sep=',')

# Load training labels
y_train = pd.read_csv("trainlabels.txt", header=None, names=['label'])


def clean_data(df):
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    numeric_df = numeric_df.dropna(axis=1, how='all')

    for column in numeric_df.columns:
        column_mean = numeric_df[column].mean()
        numeric_df[column] = numeric_df[column].fillna(column_mean)

    numeric_df = numeric_df.loc[:, numeric_df.var() != 0]

    ##numeric_df = numeric_df.iloc[:, 0:8]

    return numeric_df


X_train_cleaned = clean_data(X_train)
train_data = X_train_cleaned.to_numpy()
labels = y_train.to_numpy().ravel()



def analyze_feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    importances = rf.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': [f'{i}' for i in range(X.shape[1])],
        'Importance': importances
    })

    feature_importance = feature_importance.sort_values('Feature')

    significant_features = feature_importance[feature_importance['Importance'] > 0.02]
    print("\nSignificant features (importance > 2%):")
    print(significant_features)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature',
                data=feature_importance.sort_values('Importance', ascending=False).head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()

    return feature_importance



print("\nAnalyzing feature importance...")
feature_importance = analyze_feature_importance(train_data, labels)
threshold = 0.02 # 2% importance threshold
important_features = feature_importance[feature_importance['Importance'] > threshold]['Feature'].tolist()
train_data = train_data[:, [int(f) for f in important_features]]



X_train, X_test, y_train, y_test = train_test_split(
    train_data, labels, test_size=0.2, stratify=labels
)


scaler = MinMaxScaler((0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(input_size, 64, bias=True)
        self.layer2 = nn.Linear(64, 32, bias=True)
        self.layer3 = nn.Linear(32, 10)


        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout1d(p=0.055)

    def forward(self, x):

        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.layer3(x)

        return x


input_size = X_train.shape[1]
model = FeedForwardNN(input_size)

class_counts = np.bincount(y_train)
class_weights = 1./torch.tensor(class_counts, dtype=torch.float)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002293)

num_epochs = 250
batch_size = 8
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_x = X_train_tensor[i: i + batch_size]
        batch_y = y_train_tensor[i: i + batch_size]

        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = loss_fn(test_outputs, y_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / n_batches:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}, '
                  f'F1 Score: {f1:.4f}')


model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)

    y_true = y_test_tensor.numpy()
    y_pred = predicted.numpy()

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.4f}")


    # Calculate and print F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nWeighted F1 Score: {f1:.4f}")

#torch.save(model.state_dict(), 'ML_assignment2_model.pt')