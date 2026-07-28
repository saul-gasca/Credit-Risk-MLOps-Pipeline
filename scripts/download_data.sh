#!/bin/bash

COMPETITION="home-credit-default-risk"
DEST_DIR="datasets/raw"
FILES=("bureau.csv" "bureau_balance.csv" "previous_application.csv" "installments_payments.csv","application_train.csv")

# 1. Crea la carpeta destino si no existe
mkdir -p $DEST_DIR

# 2. Recorre el array FILES
for file in "${FILES[@]}"; do
    echo "Downloading $file..."

    # 3. Descarga el archivo individual
    kaggle competitions download -c $COMPETITION -f $file -p $DEST_DIR

    # 4. Descomprime el .zip resultante en DEST_DIR
    unzip -o "$DEST_DIR/$file.zip" -d $DEST_DIR

    # 5. Borra el .zip para no ocupar espacio doble
    rm "$DEST_DIR/$file.zip"

done

echo "Donload completed."