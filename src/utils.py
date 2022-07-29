import concurrent.futures
import glob
import os
import re
from datetime import *
from itertools import repeat

import pandas as pd
import wget
from google.cloud import storage
from sklearn.model_selection import train_test_split


def split_train_valid_test_custom(df: pd.DataFrame = None,
                                  x_col: list = ['image_id', 'image'],
                                  y_col: str = 'tag',
                                  train_split: float = 0.80,
                                  test_include: bool = False,
                                  random_seed: int = 2481):
    """
    Create custom train/valid or train/valid/test splits from a dataframe.

    :param df: Dataframe to split, defaults to None
    :param x_col: Features columns to include in the resulting dataframe beside the tag column., defaults to ['image_id', 'image']
    :param y_col: Column including the label (used to stratify the splits, i.e. have equal percentage per label), defaults to 'tag'
    :param train_split: `df_train_valid` percentage from `df` , defaults to 0.80
    :param test_include: Determines whether to return only `(df_train_valid)` if `False` or `(df_train_valid, df_test)` if `True`., defaults to False
    :param random_seed: Random seed
    """

    x_train, x_valid_test, y_train, y_valid_test = train_test_split(df[x_col], df[y_col],
                                                                    stratify=df[y_col],
                                                                    test_size=1-train_split,
                                                                    random_state=random_seed)

    df_train = x_train.assign(tag=y_train)

    if test_include:
        # split the (1-train_split) into two equal parts
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test,
                                                            stratify=y_valid_test,
                                                            test_size=0.5,
                                                            random_state=random_seed)

        df_valid = x_valid.assign(tag=y_valid)
        df_test = x_test.assign(tag=y_test).reset_index(drop=True)
    else:
        # use all the (1-train_split) as valid
        df_valid = x_valid_test.assign(tag=y_valid_test)

    # add train/valid flags
    df_train['is_valid'] = False
    df_valid['is_valid'] = True
    df_train_valid = pd.concat([df_train, df_valid]).reset_index(
        drop=True)  # merge train and valid in one df

    if test_include:
        res = (df_train_valid, df_test)
    else:
        res = (df_train_valid,)
    return(res)


def merge_label_data(df_first: pd.DataFrame = None,
                     df_second: pd.DataFrame = None,
                     image_id_column: str = 'image_id',
                     image_url_column: str = 'image',
                     unify_url_prefix: bool = True,
                     image_url_prefix: str = 'https://instance.amazonaws.com/',
                     image_path_pattern=r"[a-z]+[/\d+]+.jpeg$"):
    """
    Merge two dataframes with image info [image_id, image, etc.]

    :param df_first: First datafram, defaults to None
    :param df_second: Second dataframe to merge, defaults to None
    :param image_id_column: Column name including image id (to be used to drop duplicates), defaults to 'image_id'
    :param image_url_column: Column name including image url (to be used to docwnload the jpeg images), defaults to 'image'
    :param unify_url_prefix: Whether to unify image url prefix , defaults to True
    :param image_url_prefix: Image url prefix to use when unify_url_prefix=True, defaults to 'https://instance.amazonaws.com/'
    :param image_path_pattern: Regex pattern of image path , defaults to r"[a-z]+[/\d+]+.jpeg$" for cases where image path is like "

     :return: DataFrame including merged data.
    """

    if unify_url_prefix:
        df_first[image_url_column] = df_first[image_url_column].apply(
            lambda x: image_url_prefix+re.search(image_path_pattern, x).group())
        df_second[image_url_column] = df_second[image_url_column].apply(
            lambda x: image_url_prefix+re.search(image_path_pattern, x).group())

    # merge dataframe and drop duplicates
    df_merged = pd.concat([df_first, df_second])
    df_merged = df_merged.drop_duplicates(
        subset=image_id_column, keep='last').reset_index(drop=True)
    return(df_merged)


def create_new_data_version_gcs(df_new=None,
                                type=None,
                                bucket='bucket',
                                version_prefix='temp'):
    """
    Create a CSV from the given `df_new` under a new version in GCS. 

    :param df_new: Dataframe including data for the new version, defaults to None.
    :param type: Data type `train_valid` or `test` to, defaults to None.
    :param bucket: GCS bucket, defaults to 'gs://image-bucket'.
    :param version_prefix: version prefix.
    """

    if version_prefix == None:
        version_prefix = datetime.today().strftime('%Y%m%d')

    data_gcs_path = f'gs://{bucket}/{version_prefix}/data_{type}.csv'
    print(f'writing data to: {data_gcs_path}')
    df_new.to_csv(data_gcs_path, index=False)


def download_image(image_url=None,
                   output_dir=None):
    """
    Download images in parallel.

    :param image_url: Image url, defaults to None
    :param image_id: Image id to use as file name , defaults to None
    :param output_dir: Output directory, defaults to None
    """

    os.makedirs(output_dir, exist_ok=True)

    try:
        image_legacy_id = image_url.split('/')[-1]
        wget.download(image_url, out=output_dir)
    except Exception as e:
        print(f'Issue while downloading {image_url}--> {e}')


def download_image_concurrent(df=None, local_dir=None, image_url=None, image_id=None):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_image,
                     df[image_url], df[image_id], repeat(local_dir))


def upload_from_file_to_gcs(file_to_upload=None,
                            bucket_client=None,
                            bucket_name=None,
                            version_prefix=None):
    """
    Upload a single file to GCS bucket.

    :param file_to_upload: file to upload, defaults to None
    :param bucket_client: storage client (retrieved bucket from `storage_client.get_bucket(BUCKET_NAME)`), defaults to None
    :param bucket_name:bucket name , defaults to None
    :param version_prefix: version prefix , defaults to None

    Examples:
    >>> upload_from_file_to_gcs(file_to_upload='sample_images/6201599.jpeg',
                                bucket_client=storage_client,
                                bucket_name='image-bucket',
                                version_prefix='temp/imgs_train')
    """

    file_name = file_to_upload.split('/')[-1]
    bucket_client = bucket_client.get_bucket(bucket_name)
    blob = bucket_client.blob(f'{version_prefix}/{file_name}')
    blob.upload_from_filename(file_to_upload)
    print(f'Uploaded: {file_to_upload}')


def upload_from_dir_to_gcs(local_dir=None,
                           storage_client=storage.Client(),
                           bucket_name=None,
                           version_prefix=None):
    """
    Upload all files in `local_dir` to gcs bucket `bucket_name` under prefix `version_prefix`.

    :param local_dir: local directory including files to upload, defaults to None
    :param storage_client: storage client, defaults to None
    :param bucket_name: bucket name, defaults to None
    :param version_prefix: prefix, defaults to None

    Examples:
    >>> storage_client = storage.Client()
        upload_from_dir_to_gcs(local_dir='sample_images',
                           storage_client=storage_client,
                           bucket_name='image-bucket',
                           version_prefix='temp/imgs_train')

    """

    files_to_upload = os.listdir(local_dir)
    bucket_client = storage_client.get_bucket(bucket_name)
    for im in files_to_upload:
        blob = bucket_client.blob(f'{version_prefix}/{im}')
        blob.upload_from_filename(f'{local_dir}/{im}')
        print(f'Uploaded: {local_dir}/{im}')


def upload_from_file_to_gcs_concurrent(local_dir=None,
                                       storage_client=storage.Client(),
                                       bucket_name=None,
                                       prefix=None):
    """
    Upload all files in `local_dir` to gcs bucket `bucket_name` under prefix `version_prefix`.

    :param local_dir: local directory including files to upload, defaults to None
    :param storage_client: storage client, defaults to None
    :param bucket_name: bucket name, defaults to None
    :param version_prefix: prefix, defaults to None

    Examples:
    >>> storage_client = storage.Client()
        upload_from_file_to_gcs_concurrent(local_dir='sample_images',
                                           storage_client=storage_client,
                                           bucket_name='image-bucket',
                                           version_prefix='temp/imgs_train')
    """

    files_to_upload = glob.glob(f'{local_dir}/*')
    bucket = storage_client.get_bucket(bucket_name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(upload_from_file_to_gcs, files_to_upload,
                     repeat(bucket), repeat(prefix))
