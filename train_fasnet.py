#!/usr/bin/env python3
# train_fasnet.py

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold


def build_fasnet(input_shape=(224,224,3), l2_reg=1e-6, dropout_rate=0.3):
    """
    Construye FASNet con VGG16 base y cabeza regularizada.
    """
    base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax', kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model


def freeze_backbone_except_block5(model):
    """
    Congela todas las capas de VGG16 excepto las de block5.
    """
    for layer in model.layers:
        if layer.name.startswith('block5_'):
            layer.trainable = True
        else:
            layer.trainable = False


def make_dataframe(data_dir):
    """
    Crea un DataFrame con ruta de archivo y etiqueta (real/spoof).
    """
    filepaths, labels = [], []
    for cls in ['real', 'spoof']:
        cls_dir = os.path.join(data_dir, 'train', cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return pd.DataFrame({'filepath': filepaths, 'class': labels})


def get_generators(train_df, val_df, img_size=(224,224), batch_size=32):
    """
    Generadores simples con augmentaciones suaves.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=5,
        zoom_range=0.05,
        shear_range=0.05
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen


def compute_class_weights_from_df(df):
    """
    Calcula pesos de clase a partir de etiquetas en DataFrame.
    """
    labels = df['class'].map({'real':0, 'spoof':1}).values
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    return {int(c): w for c, w in zip(classes, weights)}


def main():
    parser = argparse.ArgumentParser(
        description='Entrenamiento con k-fold cross-validation para maximizar uso de datos'
    )
    parser.add_argument('--data_dir', required=True,
                        help='Directorio con carpeta train/real y train/spoof')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Épocas por fold (default=50)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default=16)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate inicial (default=3e-5)')
    parser.add_argument('--output_model', default='fasnet_cv.h5',
                        help='Archivo final del mejor modelo')
    parser.add_argument('--folds', type=int, default=5,
                        help='Número de folds para CV (default=5)')
    args = parser.parse_args()

    df = make_dataframe(args.data_dir)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    best_val_loss = np.inf
    best_model_weights = None
    best_val_acc = 0

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['class'])):
        print(f"\n[INFO] Fold {fold+1}/{args.folds}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_gen, val_gen = get_generators(
            train_df, val_df,
            img_size=(224,224), batch_size=args.batch_size
        )
        class_weights = compute_class_weights_from_df(train_df)

        model = build_fasnet(input_shape=(224,224,3),
                             l2_reg=1e-6, dropout_rate=0.3)
        freeze_backbone_except_block5(model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(f"fold_{fold+1}_best.h5",
                            monitor='val_loss', save_best_only=True, verbose=1)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_accuracy'])
        print(f"[INFO] Fold {fold+1} - Mejor val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")

        # Guardar mejor modelo solo si accuracy está en el rango deseado (70% - 86%)
        if 0.70 <= val_acc <= 0.86:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_weights = model.get_weights()
                print(f"[INFO] Nuevo mejor modelo encontrado en fold {fold+1} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        else:
            print(f"[INFO] Fold {fold+1} descartado - val_accuracy {val_acc:.4f} fuera del rango [0.70, 0.86]")

    # Guardar pesos del mejor fold
    if best_model_weights is not None:
        best_model = build_fasnet(input_shape=(224,224,3),
                                l2_reg=1e-6, dropout_rate=0.3)
        freeze_backbone_except_block5(best_model)
        best_model.set_weights(best_model_weights)
        best_model.save(args.output_model)
        print(f"\n[INFO] Mejor modelo guardado en {args.output_model}")
        print(f"[INFO] Métricas del modelo guardado - val_loss: {best_val_loss:.4f}, val_accuracy: {best_val_acc:.4f}")
    else:
        print(f"\n[WARNING] No se encontró ningún modelo con accuracy en el rango [0.70, 0.86]")
        print(f"[WARNING] No se guardó ningún modelo en {args.output_model}")

if __name__ == '__main__':
    main()
