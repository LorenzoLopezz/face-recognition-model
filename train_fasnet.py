#!/usr/bin/env python3
# train_fasnet.py (modificado: recomendaciones 1, 4, 5 y 9, ajustado para focal loss y sin L2)  

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
import tensorflow_addons as tfa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold


def build_fasnet(input_shape=(224,224,3), dropout_rate=0.5):
    """
    Construye FASNet con VGG16 base y cabeza regularizada (BatchNorm + Dropout).
    """
    base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs=base.input, outputs=outputs)


def freeze_backbone_except_block5(model):
    """
    Congela todas las capas de VGG16 excepto las de block5.
    """
    for layer in model.layers:
        layer.trainable = layer.name.startswith('block5_') or not layer.name.startswith('block')


def make_dataframe(data_dir):
    """
    Crea DataFrame con ruta y clase (real/spoof).
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
    Generadores con augmentaciones usando preprocess_input.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8,1.2),
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    Calcula pesos de clase.
    """
    labels = df['class'].map({'real':0, 'spoof':1}).values
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {int(c): w for c, w in zip(classes, weights)}


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento con k-fold CV')
    parser.add_argument('--data_dir', required=True, help='Directorio con train/real y train/spoof')
    parser.add_argument('--epochs', type=int, default=50, help='Épocas por fold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate inicial')
    parser.add_argument('--output_model', default='fasnet_cv.h5', help='Modelo final')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds')
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('history', exist_ok=True)

    df = make_dataframe(args.data_dir)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    best_val_loss = np.inf
    best_val_acc = 0
    best_weights = None

    # Rangos de métricas
    min_acc, max_acc = 0.80, 0.90
    min_loss, max_loss = 0, 0.15

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['class']), 1):
        print(f"[INFO] Fold {fold}/{args.folds}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_gen, val_gen = get_generators(train_df, val_df, batch_size=args.batch_size)
        class_weights = compute_class_weights_from_df(train_df)

        model = build_fasnet(dropout_rate=0.5)
        freeze_backbone_except_block5(model)

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            alpha=0.25, gamma=2.0, from_logits=False,
            reduction=tf.keras.losses.Reduction.AUTO
        )
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint('checkpoints/fold_%d_best.h5' % fold, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
            TensorBoard(log_dir=os.path.join('logs', f'fold_{fold}'))
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(f"history/history_fold_{fold}.csv", index=False)

        val_loss = hist_df['val_loss'].min()
        val_acc = hist_df['val_accuracy'].max()
        print(f"[INFO] Fold {fold} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        if min_acc <= val_acc <= max_acc and min_loss <= val_loss <= max_loss:
            if val_loss < best_val_loss:
                best_val_loss, best_val_acc = val_loss, val_acc
                best_weights = model.get_weights()
                print(f"[INFO] Nuevo mejor modelo en fold {fold}")
        else:
            print(f"[INFO] Fold {fold} descartado - métricas fuera de rango")

    if best_weights is not None:
        best_model = build_fasnet(dropout_rate=0.5)
        freeze_backbone_except_block5(best_model)
        best_model.set_weights(best_weights)
        best_model.save(args.output_model)
        print(f"[INFO] Mejor modelo guardado: {args.output_model} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
    else:
        print("[WARNING] No se guardó ningún modelo. Revisa los rangos de métricas.")

if __name__ == '__main__':
    main()
