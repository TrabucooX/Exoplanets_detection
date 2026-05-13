import lightkurve as lk
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
# import matplotlib.pyplot as plt       Use if you want to track your data by checking the plots show the correct graph.

# GLOBAL VARIABLES
NUM_INTERPOLATION = 100
MINIMUM_FOLDED_POINTS = 50
ZOOM_RANGE = 0.2
BIN_SIZE = 0.02 # Value according to formula to get a valid representation of the binned curve.


def preprocess_pipeline(initial_range, final_range):
    exoplanets = pd.read_csv("Confirmed_exoplanets_NASA_archive.csv", comment="#")
    exoplanets_labeled = exoplanets[exoplanets["koi_disposition"]!="CANDIDATE"].reset_index(drop=True)

    if final_range > len(exoplanets_labeled):
        raise ValueError(f"final_range ({final_range}) exceeds the available samples in the dataset ({len(exoplanets_labeled)}). "
        "Please provide a smaller range.")
    

    kic_cache = 0
    lc_collection = {"lcs":lk.LightCurveCollection([]), "lc_periods":[], "lc_epoch_transits":[]}
    flux_dictionary = {"label":[], "flux":[]}

    for i in tqdm(range(initial_range, final_range), colour='yellow', desc='Downloading batch of Lightcurves'):
        # Retrieving lightcurve data and searching for it
        kic_lc, kepoi_name_lc, period_lc, epoch_transit_lc, label_lc = exoplanets_labeled.loc[i, ["kepid", "kepoi_name" ,"koi_period", "koi_time0bk", "koi_disposition"]]
        quarter = int(np.ceil(period_lc/90)) # Only downloading enough quartes to get a transit.
        if kic_lc != kic_cache:
            lc_search = lk.search_lightcurve(f'KIC {kic_lc}', mission='Kepler', cadence='long', quarter=[j for j in range(1, quarter+2)])
            kic_cache = kic_lc

            try:
                lc = lc_search.download_all().stitch().remove_outliers(sigma_lower=float('inf'), sigma_upper=5)
                lc_collection.append(lc)
            except Exception as e:
                print(f"Error with KIC {kic_lc}: {e}")
                continue

        # Saving lightcurve after the if, so if the same kic is recorded, we added the same star observation but with another object.
        lc_collection["lcs"].append(lc)
        lc_collection["lc_periods"].append(period_lc)
        lc_collection["lc_epoch_transits"].append(epoch_transit_lc)

        # Computing parameters for the processing of the curve
        win = int(10 * (period_lc**0.3))
        if win % 2 == 0: win += 1
        win = max(win, 51)

        folded_lc = lc.flatten(win).fold(period_lc, epoch_time=epoch_transit_lc).truncate(-ZOOM_RANGE,ZOOM_RANGE)
        
        # Creating an interpolation flux on 100 points to store all lightcurves later with the same flux per time points.
        x_grid = np.linspace(-ZOOM_RANGE, ZOOM_RANGE, NUM_INTERPOLATION)
        # Interpolate the flux onto that grid
        if len(folded_lc) > MINIMUM_FOLDED_POINTS:
            binned_lc = folded_lc.bin(BIN_SIZE)
            fixed_flux = np.interp(x_grid, binned_lc.time.value, binned_lc.flux.value)
            flux_dictionary['label'].append(int(label_lc == "CONFIRMED"))
            flux_dictionary['flux'].append(fixed_flux)
        else:
            print(f"Skipping index {i}: Not enough points after folding.")


    return lc_collection, flux_dictionary
        
        # if i%25 == 0 and len(folded_lc)>50:
        #     fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        #     folded_lc.plot(ax=ax[0])
        #     ax[0].set_title("No bins representation")
        #     plt.plot(x_grid, fixed_flux, color='red', marker='x', label='Interpolated Grid', markersize=4)

        #     bin_size = 0.02
        #     folded_lc.bin(bin_size).plot(ax=ax[1])
        #     ax[1].set_title(f"Binned Curve with {bin_size}")

        #     plt.tight_layout()
        #     plt.show()
        #     print(f'Dip in flux of the object KIC {kic_lc}, KOI Name: {kepoi_name_lc}, which is classified as a {label_lc} (exoplanet).')

def saving_raw_flux_data(lc_collection : lk.LightCurveCollection, batch_number=1):
    data = []
    for i, lc in enumerate(lc_collection["lcs"]):
        data.append({
        'kic_id': lc.label,
        'period': lc_collection["lc_periods"][i],
        'epoch_transit': lc_collection["lc_epoch_transits"][i],
        'flux': lc.flux.value.tolist()
        })

    df = pd.DataFrame(data)
    try:
        df.to_parquet(f'data/parquet_batches/batch_{batch_number}.parquet', engine='pyarrow', compression='snappy')
        print('Raw values have been stored correctly.')
    except Exception as e:
        print('Something has failed during the saving process, please try again later.')

def saving_processed_flux_data(flux_dictionary : dict, batch_number = 1):
    df_new = pd.DataFrame(flux_dictionary['flux'])
    df_new.columns = [f'flux_{i}' for i in range(NUM_INTERPOLATION)]
    df_new['label'] = flux_dictionary['label']

    # Computing skewness, kurtosis and minimum of our interpolated-flux points.
    df_subset = df_new.iloc[:, :NUM_INTERPOLATION-1]
    df_new["skew"] = df_subset.skew(axis=1).fillna(1.0)
    df_new["kurtosis"]  = df_subset.kurt(axis=1).fillna(1.0)
    df_new["mins"] = df_subset.min(axis=1)
    df_new.dropna(axis=0, inplace=True)

    try:
        df_new.to_parquet(f"data/interpolated_batches/exoplanets_interpolated_flux_data_batch_{batch_number}.parquet",
                          engine='pyarrow', compression='snappy', index=False)
        print('Interpolated flux values have been stored correctly.')
    except Exception as e:
        print('Something has failed during the saving process, please try again later.')

if __name__ == "__main__":
    print("Starting illustrative preprocessing batch...")
    
    # Run a small sample (e.g., 5 curves)
    lcs, flux_dict = preprocess_pipeline()
    
    saving_processed_flux_data(flux_dict, batch_number="test")
    print("Test batch complete. Check the 'data/' folder.")
