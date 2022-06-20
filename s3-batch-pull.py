import fsspec
import xarray as xr
from tqdm import tqdm
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from fsspec.implementations.cached import SimpleCacheFileSystem
# import worker


def fn(fp, fs=None):
    return xr.open_dataset(fs.open(fp), engine='h5netcdf', chunks={"Rad": (30, 50)}, drop_variables=worker.drop_var)


def run_tqdm(func, data, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    tmp = []
    for result in tqdm(pool.imap(func=func, iterable=data), total=len(data)):
        tmp.append(result)

    return tmp


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    # Define values
    bucket_name = 'noaa-goes17'  # West
    product_name = 'ABI-L1b-RadC'
    year = 2022
    band = 7

    # Get s3 path
    print("Getting s3 path")
    glob_path = []

    for day_of_year in range(70, 101):
        for hour in range(4, 10):  # 4 -> 9 for CST,
            glob_path.append(
                f'{bucket_name}/{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}*.nc')

    # Define fsspec
    fs = fsspec.filesystem("s3", anon=True)
    fs = SimpleCacheFileSystem(
        fs=fs,
        cache_storage="./cache_storage",
        expiry_time=None,
        same_names=True,
    )

    print("Open s3")
    files = []
    for path in glob_path:
        for fp in fs.glob(path):
            files.append(fp)

    func = partial(fn, fs=fs)

    tmp = []
    for fp in tqdm(files):
        def open_dataset(f):
            return xr.open_dataset(fs.open(f), engine='h5netcdf', chunks={"Rad": (30, 50)}, drop_variables=worker.drop_var)
        tmp.append(open_dataset(fp))
        del open_dataset

    # tmp = run_tqdm(func, files, 8)
    ds_C07 = xr.concat(tmp, dim="new_dim")
    new_dim = ds_C07.dims["new_dim"]
    print(f"ds_C07 dim: {new_dim}")

    fig, ax = plt.subplots()

    ims = []
    for i in tqdm(range(new_dim)):
        ax.axis('off')
        im = ax.imshow(ds_C07.Rad[i].values, cmap='hsv', animated=True)
        if i == 0:
            ax.imshow(ds_C07.Rad[0].values, cmap='hsv')
        ims.append([im])

    ani = ArtistAnimation(fig, ims, interval=50, blit=True,
                          repeat_delay=1000)
    plt.axis('off')
    plt.show()
    ani.save('GOES17_C07_70_100.gif', writer='imagemagick', fps=120)
