"""
Ejecuta el pipeline completo (Pasos 1 a 6).
"""
import polars as pl

from config import rutas as cfg
from src.cleaning import preprocess                           
from src.models import correlacion, normalizacion, pca_model 

#Dataset final en data-input-model
PREPARED_FILE = cfg.DATA_INPUT_MODEL / "household_prepared.parquet"


def main():
    """Corre todos los pasos en orden y guarda el dataset final."""
    cfg.DATA_INPUT_MODEL.mkdir(parents=True, exist_ok=True)  #Crea la carpeta de salida

    #Carga, inspección y limpieza 
    preprocess.run()
    df_limpio = pl.read_parquet(preprocess.CLEAN_FILE)  #Se relee el limpio desde disco

    #Normalización con parámetros ajustados solo con el tramo de entrenamiento
    df_norm, params = normalizacion.normalize(df_limpio)
    print(f"[Paso 3] método: {params['method']} | ajustado con datos hasta {params['ajustado_con_datos_hasta']}")
    #Comprobación en train las medias deben quedar aprox 0 y las desviaciones aprox 1
    print(f"[Paso 3] comprobación en train: {normalizacion.describe_after(df_norm, cfg.NUMERIC_COLS)}")

    #Pearson y selección de variables
    conservadas, _ = correlacion.run(df_norm)  #Lista de variables que no son redundantes

    #PCA sobre las variables que se conservaron
    df_pca = pca_model.run(df_norm, conservadas)

    #Dataset final con variables conservadas, componentes y split
    componentes = [c for c in df_pca.columns if c.startswith("PC")]
    final = df_pca.select([cfg.DATETIME_COL, *conservadas, *componentes, "split"])
    #Float32 reduce a la mitad el tamaño del archivo
    final = final.with_columns(pl.col([*conservadas, *componentes]).cast(pl.Float32))
    final.write_parquet(PREPARED_FILE, compression="zstd")
    print(f"[Paso 6] {PREPARED_FILE.name}: {final.shape[0]:,} filas x {final.shape[1]} columnas -> {final.columns}")


if __name__ == "__main__":
    main()