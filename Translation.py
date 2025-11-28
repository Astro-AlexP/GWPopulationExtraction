import bilby
import numpy as np
import pandas as pd
import glob

resultPath = "./Bilby_Data/*/*_result.json"

pe_files = np.sort(glob.glob(resultPath))

for file in pe_files:
    result = bilby.read_in_result(filename=file)

    posterior_df = pd.DataFrame(result.posterior)

    event = file[20:28]

    for i, col in enumerate(posterior_df.columns):
        if i == 0:
            posterior_df[col].to_hdf("Data/" + event + "_data.hdf5", key=col, mode='w')

        else:
            posterior_df[col].to_hdf("Data/" + event + "_data.hdf5", key=col, mode='a')

    file_path = "Data/" + event + "_data.hdf5"

    print(f"DataFrame successfully written to {file_path} with columns as keys.")
    # Optional: Verify the keys in the HDF file
    with pd.HDFStore(file_path, mode='r') as store:
        print(f"Keys in HDF file: {store.keys()}")