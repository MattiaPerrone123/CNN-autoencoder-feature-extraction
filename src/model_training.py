import torch
import numpy as np
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from .metrics import get_iou_train, get_dice, get_mean_iou_train
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier


def initialize_model(img_size, in_channels, out_channels, feature_size, lr):
    """Initialize the model, optimizer, and loss function"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    ).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn=nn.BCELoss()
    return model, optimizer, loss_fn


def train_and_validate(model, training_loader, testing_loader, optimizer, criterion, num_epochs, device, save_path='best_model_weights.pth', load_weights=False):
    """Train and validate the model, or load pre-trained weights if specified"""
    if load_weights and save_path and torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(save_path))
            model.eval()
            print(f"Model weights loaded from {save_path}. Skipping training.")
            return {"model": model, "metrics": None}
        except FileNotFoundError:
            print(f"No weights found at {save_path}. Proceeding with training.")
    
    best_iou=0.0
    metrics={'train_losses': [], 'train_ious': [], 'train_dices': [], 'test_ious': [], 'test_dices': []}

    for epoch in range(num_epochs):
        train_running_loss=0.0
        train_iou=0.0
        train_dice=0.0

        model.train()
        for labels, images in training_loader:
            images, labels=images.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits=model(images)
            logits=torch.sigmoid(logits)
            loss=criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_running_loss+=loss.item()
            train_iou+=get_iou_train(logits, labels)
            train_dice+=get_dice(logits, labels)

        model.eval()
        test_iou, test_dice=0.0, 0.0
        with torch.no_grad():
            for labels, images in testing_loader:
                images, labels=images.float().to(device), labels.float().to(device)
                logits=torch.sigmoid(model(images))
                test_iou+=get_iou_train(logits, labels)
                test_dice+=get_dice(logits, labels)

        train_loss=train_running_loss/len(training_loader)
        train_iou/=len(training_loader)
        train_dice/=len(training_loader)
        test_iou/=len(testing_loader)
        test_dice/=len(testing_loader)

        metrics['train_losses'].append(train_loss)
        metrics['train_ious'].append(train_iou)
        metrics['train_dices'].append(train_dice)
        metrics['test_ious'].append(test_iou)
        metrics['test_dices'].append(test_dice)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Train IoU: {train_iou:.2f} | Train Dice: {train_dice:.2f} | Test IoU: {test_iou:.2f} | Test Dice: {test_dice:.2f}")

        if test_iou>best_iou:
            best_iou=test_iou
            torch.save(model.state_dict(), save_path)

    return {"model": model, "metrics": metrics}


def initialize_training(autoencoder, lr=0.001):
    """Initialize the autoencoder, loss function, optimizer and device"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=autoencoder.to(device)
    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer, device


def train_one_epoch(model, criterion, optimizer, training_loader, device):
    """Train the autoencoder for one epoch"""
    running_loss=0.0
    stop_training=False
    last_inputs, last_outputs=None, None

    for i, data in enumerate(training_loader):
        inputs=data[0].float().to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        outputs=torch.sigmoid(outputs)
        loss=criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

        last_inputs, last_outputs=inputs, outputs

        if get_mean_iou_train(inputs, outputs)>0.995:
            stop_training=True
            break

    return running_loss, stop_training, last_inputs, last_outputs


def train_model(autoencoder, training_loader, num_epochs=1000, lr=0.001, save_path="model_weights.pth"):
    """Train the autoencoder model."""
    model, criterion, optimizer, device=initialize_training(autoencoder, lr)
    last_inputs, last_outputs=None, None

    for epoch in range(num_epochs):
        running_loss, stop_training, last_inputs, last_outputs=train_one_epoch(
            model, criterion, optimizer, training_loader, device
        )
        if stop_training:
            print("Training IoU higher than 0.995")
            break

        print(f"Epoch {epoch+1}, Loss: {round(running_loss/len(training_loader),3)}, IoU: {round(get_iou_train(last_inputs, last_outputs),2)}")

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    print('Finished Training')
    return model


def load_model(autoencoder, load_path):
    """Load model weights from a file"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=autoencoder.to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model weights loaded from {load_path}")
    return model


def cross_validation_model_evaluation(latent_features, target_data, feature_columns, type_f1_score, random_state=29, n_splits=5):
    """Perform stratified K-fold cross-validation to evaluate F1 and AUC scores."""
    f1_scores_micro_tot=[]
    auc_scores_tot=[]

    scaler=StandardScaler()
    scaled_features=scaler.fit_transform(latent_features)
    kf=StratifiedKFold(n_splits=n_splits, shuffle=False)

    for feature in feature_columns:
        y=target_data[feature].to_numpy()
        label_encoder=LabelEncoder()
        y_transformed=label_encoder.fit_transform(y)
        xgb=XGBClassifier(n_estimators=200, random_state=random_state)
        
        f1_scores_micro=[]
        auc_scores=[]
        
        for train_index, test_index in kf.split(scaled_features, y_transformed):
            X_train, X_test=scaled_features[train_index], scaled_features[test_index]
            y_train, y_test=y_transformed[train_index], y_transformed[test_index]
            xgb.fit(X_train, y_train)
            y_pred=xgb.predict(X_test)
            y_pred_proba=xgb.predict_proba(X_test)[:, 1]
            f1_micro=f1_score(y_test, y_pred, average=type_f1_score)
            f1_scores_micro.append(f1_micro)
            auc_score=roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc_score)
        
        f1_scores_micro_tot.append(f1_scores_micro)
        auc_scores_tot.append(auc_scores)

        print(f"Feature: {feature}")
        print(f"Mean F1 Score (Micro): {np.mean(f1_scores_micro)}")
        print(f"Mean AUC Score: {np.mean(auc_scores)}")
        print("\n"+"="*50+"\n")
    
    return f1_scores_micro_tot, auc_scores_tot
