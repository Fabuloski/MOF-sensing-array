import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, auc
    from sklearn.preprocessing import PowerTransformer
    from sklearn.decomposition import PCA
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec
    return (
        ConfusionMatrixDisplay,
        KNeighborsClassifier,
        Line2D,
        LinearRegression,
        LinearSegmentedColormap,
        ListedColormap,
        PCA,
        Path,
        PowerTransformer,
        auc,
        gridspec,
        make_axes_locatable,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(plt):
    plt.rcParams.update({"font.size": 18})
    return


@app.cell
def _():
    from aquarel import load_theme

    theme = load_theme("boxy_light")
    theme.apply()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helpers to read in response data""")
    return


@app.cell
def _(Path, np, pd):
    """
        _find_ppm_sheet(filname, ppm)

    read in excel file and check the sheet names which include the ppm number.
    80 ppm is always the first sheet in data, so only called when ppm != 80
    """

    def _find_ppm_sheet(filename, ppm):
        xlfl = pd.ExcelFile(filename)
        sheet_names = xlfl.sheet_names
        target_sheet = [sheet for sheet in sheet_names if str(ppm) + "ppm"  in sheet.replace(" ", "")][0]
        return target_sheet

    """
        _find_header_row(filename, search_terms=['Time', 's'])

    read in excel file and check first ten rows for search terms.
    return the first row in which a search term appears.
    if not found, return None.
    """
    def _find_header_row(filename, ppm_sheet=0, search_terms=['Time', 's']):
        for i in range(10):  # Check first 10 rows
            try:
                df = pd.read_excel(filename, sheet_name=ppm_sheet, header=i, nrows=1)
                for search_term in search_terms:
                    if search_term in df.columns:
                        return i
            except:
                pass
        return None  # If header not found

    """
        read_data(cof, gas, carrier, ppm)

    read in the sensor response data for a given COF exposed to a
    given gas with a given carrier at a given concentration.
    returns list of pandas data frames with this data. (may be replicates)
    each data frame has two columns: time, DeltaG/G0.

    note: this is complicated because there are two formats for a given
    cof, gas, carrier, ppm:
    (1) multiple replicates in the same file
    (2) multiple replicates in separate files
    """
    def read_data(MOF, gas, ppm, time_adjust=0):

        ppms = [5, 10, 20, 25, 40, 80]

        path = Path.cwd().joinpath("data", gas).rglob("*.xlsx")
        # folders contain multiple excel files, so extract relevant
        files = [file for file in path if MOF in file.name]

        # extract data from Excel files in list
        dfs = []
        for filename in files:
            ppm_sheet = None
            if ppm in ppms:
                ppm_sheet = _find_ppm_sheet(filename, ppm)
                # read in file (need to find header row; not consistent)
                header_row = _find_header_row(filename, ppm_sheet)
                df = pd.read_excel(filename, sheet_name=ppm_sheet, header=header_row)

            else:
                raise Exception("PPM not supported.")

            #    only keep a subset of the cols (Time and (perhaps multiple) with muA's)
            ids_cols_keep = df.columns.str.contains('A', na=False) | (df.columns == 's')
            # exposure time begins at 780s, ends 2580s
            start_index = df.index[df['s'] == 780 + time_adjust].tolist()[0]
            end_index = df.index[df['s'] == 2580 + time_adjust].tolist()[0]
            df = df.loc[start_index:end_index, df.columns[ids_cols_keep]]

            # check time is sliced properly
            assert df.iloc[0]["s"] == 780.0 + time_adjust
            assert df.iloc[-1]["s"] == 2580.0 + time_adjust
            # reshift time
            df["s"] = df["s"] - (780.0 + time_adjust)

            # drop columns with missing values
            df = df.dropna(axis='columns')

            df.reset_index(drop=True, inplace=True)

            # separate replicates into differente dataframes and append to dfs
            for i in df.columns:
                if 'A' in i and not np.all(df[i] == 0):
                    data_rep = df[['s', i]]
                    G0 = df[i].iloc[0]
                    # replace muA column with -deltaG/G0 calculation: -ΔG/G0 = -(muA - G0)/G0 * 100
                    data_rep.loc[:, i] = 100 * (-(data_rep.loc[:, i] - G0) / G0)
                    data_rep = data_rep.rename(columns={i: "-ΔG/G0"})
                    dfs.append(data_rep)

        return dfs
    return (read_data,)


@app.cell
def _(read_data):
    read_data("NiPc-Cu", "NO", 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Helper function to run linear regression""")
    return


@app.cell
def _(LinearRegression):
    """
        linear_regression(df, ids_split)

    perform linear regression on df[ids_split]:
    ΔG/G0 = m * t + b

    # arguments:
    * df := dataframe of a single partition of sensor_response data
    * ids_split := indices of response data partition

    # output: dict of:
    * coef := coefficient from linear regression
    * r2 := r2 score
    * ids_split
    """
    def linear_regression(df, ids_split):
        X = df.loc[ids_split, "s"].to_numpy().reshape(-1, 1)
        y = df.loc[ids_split, "-ΔG/G0"].to_numpy()

        reg = LinearRegression().fit(X, y)

        r2 = reg.score(X, y)

        slope = reg.coef_[0]
        intercept = reg.intercept_

        return {'slope': slope, 'r2': r2, 'ids_split': ids_split, 'intercept': intercept}
    return (linear_regression,)


@app.cell
def _(auc, linear_regression, np, plt, read_data):
    class SensorResponse:
        def __init__(self, MOF, gas, ppm, replicate_id, time_adjust=0):
            self.MOF = MOF
            self.gas = gas
            self.ppm = ppm
            self.replicate_id = replicate_id
            self.time_adjust = time_adjust

            try:
                self.data = read_data(MOF, gas, ppm, time_adjust=self.time_adjust)[replicate_id]
            except IndexError:
                print(f"Error: replicate_id {replicate_id} does not exist for {gas}  in {MOF} at {ppm} ppm.")

            # store features
            self.slope_info = None
            self.saturation = None
            self.auc = None

        """
        compute_initial_slope(self, partition_size, total_time_window, mse_bound)
        estimate initial slope of data.

          arguments:
              * max_time := indicates the window of time from 0 to max_time to partition data
              * n_partitions := number of partitions
              * r2_bound := bound on acceptable r-squared values from linear regression
        """
        def compute_initial_slope(self, n_partitions=15, max_time=750.0, r2_bound=0):
            early_df = self.data[self.data["s"] < max_time]

            # partition data indices
            ids_splits = np.array_split(early_df.index, n_partitions)

            # create list of regression on each partition of data which satisfy the mean_squared error bound
            regression_data = [linear_regression(early_df, ids_split) for ids_split in ids_splits]
            # filter according to r2
            regression_data = list(filter(lambda res: res['r2'] > r2_bound, regression_data))

            if len(regression_data) == 0:
                raise Exception("Data has no initial slopes that satisfy r2 bound.")

            # find index of max absolute value of linear regression coefficients
            id_initial_slope = np.argmax([np.abs(rd['slope']) for rd in regression_data])

            # return regression_data which contains the initial slope
            self.slope_info = regression_data[id_initial_slope]
            return self.slope_info

        def compute_saturation(self, n_partitions=100):
            ids_splits = np.array_split(self.data.index, n_partitions)

            # get mean over partitions
            means = [np.mean(self.data.iloc[ids_split]['-ΔG/G0']) for ids_split in ids_splits]
            id_max_magnitude = np.argmax(np.abs(means))

            self.saturation = means[id_max_magnitude]
            return self.saturation

        def compute_features(self, n_partitions_saturation=100, n_partitions_slope=15, r2_bound_slope=0):
            self.compute_saturation(n_partitions=n_partitions_saturation)
            self.compute_initial_slope(n_partitions=n_partitions_slope, r2_bound=r2_bound_slope)
            self.compute_area_under_response_curve()

        # compute area under curve for each GBx DeltaG/G0 using sklearn auc
        def compute_area_under_response_curve(self):
            self.auc = auc(self.data["s"], self.data['-ΔG/G0'])
            return self.auc

        def viz(self, save=True): # viz the data along with the response features or function u fit to it.
            if self.slope_info == None or self.saturation == None:
                raise Exception("Compute features first.")

            fig, ax = plt.subplots()

            plt.xlabel("time [s]")
            plt.ylabel(r"$\Delta G/G_0$")

            # plot raw response data
            plt.scatter(self.data['s'], self.data['-ΔG/G0'])

            ###
            #   viz features
            ###
            # saturation
            plt.axhline(self.saturation, linestyle='-', color="gray")

            # slope
            t_start = self.data.loc[self.slope_info["ids_split"][0], 's']
            t_end = self.data.loc[self.slope_info["ids_split"][-1], 's']
            plt.plot(
                [t_start, t_end],
                self.slope_info["slope"] * np.array([t_start, t_end]) + self.slope_info["intercept"],
                color='orange'
            )

            all_info = "{}_{}_{}ppm_{}".format(self.MOF, self.gas, self.ppm, self.replicate_id)
            plt.title(all_info)

            if save:
                plt.savefig("responses/featurized_{}.png".format(all_info), format="png")
            plt.show()

    return (SensorResponse,)


@app.cell
def _(SensorResponse):
    # Test the SensorResponse class initial_slope function
    _sensor_response = SensorResponse("CuPc-O-Ni", "NO", 80, 1)
    _sensor_response.compute_features(n_partitions_slope=10)
    _sensor_response.viz(save=True)
    return


@app.cell
def _():
    gases = ['CO', 'H2S', 'NO']
    MOFs = ["NiPc-O-Ni", "NiPc-O-Cu", "NiPc-O-Zn", "CuPc-O-Ni", "CuPc-O-Cu","CuPc-O-Zn", "ZnPc-O-Ni", "ZnPc-O-Cu", "ZnPc-O-Zn"]
    features = ['AUC', 'slope', 'saturation']
    ppms = [80, 40, 25, 20, 10, 5]
    return MOFs, features, gases, ppms


@app.cell
def _():
    # Read data from existing data in csv or loop through raw data?
    read_data_from_file = False
    return (read_data_from_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Loop through raw data to compute all sensor responses""")
    return


@app.cell
def _(MOFs, SensorResponse, gases, ppms, read_data_from_file):
    # list for data, will append MOF, gas, and features of each sensor_response
    raw_data = []
    for gas in gases:
        for MOF in MOFs:
            for ppm in ppms:
                for rep_id in range(8):
                    if read_data_from_file:
                        continue
                    try:
                        sensor_response = SensorResponse(MOF, gas, ppm, rep_id)
                        sensor_response.compute_features()
                        sensor_response.viz(save=True)
                        raw_data.append([MOF, gas, ppm, rep_id, sensor_response.slope_info['slope'],
                                    sensor_response.saturation, sensor_response.auc]) # be consistent with features above

                    except (AttributeError, Exception):
                        pass
    return (raw_data,)


@app.cell
def _(pd, raw_data, read_data_from_file):
    # Put list of data into dataframe
    if not read_data_from_file:
        prelim_data = pd.DataFrame(raw_data, columns=['MOF', 'gas', 'ppm', 'rep_id', 'slope', 'saturation', 'AUC'])

        prelim_data # b/c we'll make adjustements later.
    return (prelim_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Time adjustment for sensor delay or human error""")
    return


@app.cell
def _(SensorResponse):
    # input data, experiment, and slope partition adjustment, output: dataframe and viz with adjusted slope feature
    def make_adjustment(
        prelim_data, MOF, gas, ppm, rep_ids, 
        n_partitions_slope_adj=15, n_partitions_saturation_adj=100, time_adjust=0
    ):
        for rep_id in rep_ids:
            try: 
                sensor_response = SensorResponse(MOF, gas, ppm, rep_id, time_adjust=time_adjust)
                sensor_response.compute_features(n_partitions_slope=n_partitions_slope_adj, 
                                                 n_partitions_saturation=n_partitions_saturation_adj)
                sensor_response.viz(save=True)
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['gas']==gas)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'slope'] = sensor_response.slope_info['slope']
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['gas']==gas)
                                    & (prelim_data['rep_id']==rep_id), 'AUC'] = sensor_response.auc
                prelim_data.loc[(prelim_data['MOF']==MOF)
                                    & (prelim_data['gas']==gas)
                                    & (prelim_data['ppm']==ppm)
                                    & (prelim_data['rep_id']==rep_id), 'saturation'] = sensor_response.saturation
            except:
                pass
        return prelim_data
    return (make_adjustment,)


@app.cell
def _(make_adjustment, pd, prelim_data, read_data_from_file):
    # do all of these in one cell.
    if not read_data_from_file:
        data = prelim_data.copy()
        make_adjustment(data, MOF='CuPc-O-Cu', gas='CO', ppm=20, rep_ids=[0, 1, 3], time_adjust=40)
        make_adjustment(data, MOF='CuPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=80)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='H2S', ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=50)
        make_adjustment(data, MOF='CuPc-O-Zn', gas='H2S', ppm=10, rep_ids=[0, 1, 2, 3], time_adjust=120)
        make_adjustment(data, MOF='CuPc-O-Zn', gas='H2S', ppm=40, rep_ids=[0, 1, 2], time_adjust=40)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=20, rep_ids=[0, 2, 3], time_adjust=80)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2], time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=80, rep_ids=[0, 1, 2], time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=300)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='H2S', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=40)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='H2S', ppm=80, rep_ids=[0, 1, 2, 3], time_adjust=30)
        make_adjustment(data, MOF='ZnPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=80)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=300, n_partitions_slope_adj=5)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='CO', ppm=80, rep_ids=[0, 1, 2, 3],  time_adjust=80)
        make_adjustment(data, MOF='ZnPc-O-Zn', gas='H2S', ppm=10, rep_ids=[0, 1, 2, 3], time_adjust=200)
        make_adjustment(data, MOF='ZnPc-O-Zn', gas='H2S', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=20, rep_ids=[0, 2], time_adjust=80)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=20, rep_ids=[3], time_adjust=180)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=20, rep_ids=[1], n_partitions_slope_adj=2, time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2], time_adjust=70)
        make_adjustment(data, MOF='NiPc-O-Cu', gas='CO', ppm=80, rep_ids=[0, 1, 2], time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=400, n_partitions_slope_adj=3)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='CO', ppm=20, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='NiPc-O-Zn', gas='CO', ppm=80, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='NiPc-O-Ni', gas='CO', ppm=80, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3, time_adjust=50)
        make_adjustment(data, MOF='NiPc-O-Ni', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], n_partitions_slope_adj=3)
        make_adjustment(data, MOF='NiPc-O-Ni', gas='CO', ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=600, n_partitions_slope_adj=2)
        make_adjustment(data, MOF='CuPc-O-Ni', gas='CO', ppm=20, rep_ids=[0, 1, 2], time_adjust=-200)
        make_adjustment(data, MOF='CuPc-O-Zn', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='CuPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='CuPc-O-Cu', gas='CO', ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=600)
        make_adjustment(data, MOF='ZnPc-O-Cu', gas='CO', ppm=80, rep_ids=[0, 1, 2, 3], time_adjust=20)
        make_adjustment(data, MOF='ZnPc-O-Cu', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='ZnPc-O-Cu', gas='CO', ppm=20, rep_ids=[0, 1, 2, 3], time_adjust=800)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='CO', ppm=80, rep_ids=[0, 1, 2, 3], time_adjust=100)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='CO', ppm=40, rep_ids=[0, 1, 2, 3], time_adjust=300, n_partitions_slope_adj=3)
        make_adjustment(data, MOF='ZnPc-O-Ni', gas='CO', ppm=20, rep_ids=[0, 1, 2], time_adjust=700, n_partitions_slope_adj=3)
        make_adjustment(data, MOF='ZnPc-O-Zn', gas='CO', ppm=80, rep_ids=[0, 1, 2], n_partitions_slope_adj=3)
        # save
        data.to_csv("responses.csv")
    else:
        data = pd.read_csv("responses.csv") # this is adjusted.
        data.drop(columns=['Unnamed: 0'], inplace=True) # remove index column, artifact of reading in
    return (data,)


@app.cell(hide_code=True)
def _():
    # Assemble and standardize complete array response vectors
    return


@app.cell
def _(MOFs, features):
    feature_col_names = [MOF + " " + feature for MOF in MOFs for feature in features]
    return (feature_col_names,)


@app.cell
def _(MOFs, feature_col_names, features, gases, np, pd, ppms):
    def assemble_array_response(data, gases=gases, ppms=ppms, MOFs=MOFs, n_replicates=7, features=features, 
                                feature_col_names=feature_col_names):
        #  matrix will store response features.
        #  col = sensor array response vector
        #  row = particular response feature for a particular MOF (9. 3 MOFs x 3 feature each)
        #  loop through data to build matrix column by column (technically row by row and then transpose)

        matrix = []
        experiments = [] # List which will store experiment setup for each array column
        for gas in gases:
            for ppm in ppms:
                for rep in range(n_replicates):
                    col = []
                    experiment = {'ppm': ppm,
                                'rep_id': rep,
                                 'gas' : gas}
                    for MOF in MOFs:
                        for (i, feature) in enumerate(features):
                            try:
                                val = data.loc[(data['MOF']==MOF)
                                                & (data['gas']==gas)
                                                & (data['ppm']==ppm)
                                                & (data['rep_id']==rep)][feature]
                                assert len(val) <= 1, "more than one instance"

                                col.append(val.iloc[0])
                            except (IndexError, KeyError):
                                pass

                    # only append column if entire array response exists
                    if len(col) == len(MOFs) * len(features):
                        matrix.append(col)
                        experiments.append(experiment)
                    else:
                        print("No complete array for experiment: ", experiment)
     # join experiments and responses in one combo data frame.
        matrix = np.array(matrix)
        response_array = pd.DataFrame(matrix, columns=feature_col_names)
        combo_df = pd.DataFrame(experiments).join(response_array)


        return combo_df
    return (assemble_array_response,)


@app.cell
def _(assemble_array_response, data):
    combo_df = assemble_array_response(data)
    combo_df
    return (combo_df,)


@app.cell
def _(PowerTransformer, combo_df, feature_col_names):
    transformed_combo_df = combo_df.copy()
    transformed_combo_df[feature_col_names] = PowerTransformer().fit_transform(transformed_combo_df[feature_col_names])
    transformed_combo_df
    return (transformed_combo_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# create heatmap of sensor feature values""")
    return


@app.function
def gases_to_pretty_name(gases):
    sub = {"CO" : "CO", "NO" :"NO", "H2S" : "H$_2$S"}
    pretty_name = []
    if isinstance(gases, str):
        return sub[gases]
    for gas in gases:
        pretty_name.append(sub[gas])
    return pretty_name


@app.cell
def _(
    LinearSegmentedColormap,
    MOFs,
    feature_col_names,
    features,
    gases,
    make_axes_locatable,
    np,
    plt,
    sns,
):
    def plot_heatmap(transformed_combo_df):
        RdGn = cmap = LinearSegmentedColormap.from_list("mycmap", ["red", "white", "green"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios':[4, 1]})

        # font size
        fs = 25
        yticklabels = features * 9

        # create heatmap
        heat_matrix_plot = transformed_combo_df[feature_col_names].T
        heat = sns.heatmap(heat_matrix_plot, cmap="coolwarm", center=0, yticklabels=yticklabels,
                           vmin=-2, vmax=2, square=True, ax=ax1, cbar=False)
        ax1.grid(False)
        # create a new axes for the colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=1.9)  # increase pad to move colorbar further right

        # add colorbar to the new axes
        cbar = fig.colorbar(heat.collections[0], cax=cax)

        # adjust colorbar ticks
        cbar.ax.tick_params(labelsize=fs)
        cbar.ax.minorticks_off()
        cbar.set_ticks([-2, -1, 0, 1, 2])

        # add colorbar label
        cbar.set_label(label='transformed\nresponse', size=fs)

        # label the gases:
        colordict = {'CO': 'grey', 'H2S': 'goldenrod', 'NO': 'purple'} # colors for different gas types


        n_exp = len(transformed_combo_df)
        # count number of experiments for each type of gas
        gas_counts = {gas: sum(transformed_combo_df.gas == gas) for gas in gases}

        ax1.annotate("CO", color=colordict["CO"], xy=((gas_counts['CO']/2 ) / n_exp, 1.04), xycoords='axes fraction',
                        fontsize=fs, ha='center', va='bottom',
                        bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                        arrowprops=dict(arrowstyle='-[, widthB=2.9, lengthB=.8', lw=2, color='k'))

        ax1.annotate("H$_2$S", color=colordict["H2S"], xy=((gas_counts['CO'] + gas_counts['H2S'] / 2 ) / n_exp, 1.04), 
                     xycoords='axes fraction', fontsize=fs, ha='center', va='bottom',
                     bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=4.3, lengthB=.8', lw=2, color='k'))

        ax1.annotate("NO", color=colordict["NO"], xy=((gas_counts['CO'] + gas_counts['H2S'] + gas_counts['NO'] / 2) / n_exp, 1.04), 
                     xycoords='axes fraction', fontsize=fs, ha='center', va='bottom',
                     bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                     arrowprops=dict(arrowstyle='-[, widthB=3.25, lengthB=.8', lw=2, color='k'))

        # label the MOFs:
        for (i, MOF) in enumerate(MOFs[::-1]):
            point = (1.5 + 3 * i)
            ax1.annotate(MOF, xy=(1.01, point / (3 * len(MOFs))), xycoords='axes fraction',
                        fontsize=fs, ha='left', va='center',
                        bbox=dict(boxstyle='square', ec='white', fc='white', color='k'),
                        arrowprops=dict(arrowstyle=']- ,widthA=1.08, lengthA=1, angleA=180', lw=2, color='k'))

        ax1.set_xticks([])
        ax1.minorticks_off()
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=fs)

        # create scatter ppm plot
        colorlist = [colordict[gas] for gas in transformed_combo_df.gas] # create list to assign color to each ppm data point

        ax2.bar(x=np.arange(0, n_exp, 1), height=transformed_combo_df['ppm'], color=colorlist)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylabel("concentration\n[ppm]\nin dry N$_2$", fontsize=fs)
        ax2.tick_params(axis='both', which='both', labelsize=fs)
        ax2.set_xticks(ticks=np.arange(0, n_exp, 1), labels=[])

        # adjust the position of ax2 to align with ax1
        plt.subplots_adjust(hspace=0.01)
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos1.x0, pos2.y0, pos1.width - 0.15, pos2.height])


        # make ppm plot nice
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(left=-0.22)
        ax2.set_ylim(top=90, bottom=-0.1)

        ax2.grid(axis='x', color='grey')
        ax2.set_yticks(ticks=[80,40,0])
        ax2.minorticks_off()
        plt.savefig("heatmap.png", bbox_inches='tight', pad_inches=0.5)
        return plt.show()
    return (plot_heatmap,)


@app.cell
def _(plot_heatmap, transformed_combo_df):
    plot_heatmap(transformed_combo_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3 x 3 plots""")
    return


@app.cell
def _(data):
    data["M1"] = data.MOF.apply(lambda x : x[:2]).copy()
    data["M2"] = data.MOF.apply(lambda x : x.split("-")[-1]).copy()
    return


@app.cell
def _(PowerTransformer, cmap, data, gases, gridspec, np, pd, plt, sns):
    def draw_3x3_response(ppm, feature, data=data, gases=gases):
        feature_to_expressive_name = {"AUC" : "area under the curve\n(AUC)",
                                      "slope" : "initial rate of response\n(slope)",
                                      "saturation" : "maximum response\n(saturation)"}

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(3, 1, hspace=0.1)
        sub_data = data[(data["ppm"] == ppm)]
        # data to help scale vizualization
        transformed_feature = np.hstack(PowerTransformer().fit_transform(pd.DataFrame(sub_data[feature])))
        sub_data.loc[sub_data.index, feature] = transformed_feature

        clip = np.max(np.abs(sub_data[feature]))
        for i, gas in enumerate(["H2S", "NO", "CO"]):
            subset_data = sub_data[(sub_data["gas"] == gas)]
            subset_data = subset_data.groupby(["M1", "M2"]).mean(feature)
            subset_data = subset_data.pivot_table(index="M1", columns="M2")
            subset_data = subset_data.loc[["Ni", "Cu", "Zn"]]

            related_cols = [col for col in subset_data.columns if feature in col]
            subset_data = subset_data[related_cols]
            subset_data = subset_data.iloc[:, [1, 0, 2]] # ["Cu", "Ni", "Zn"] -> ["Ni", "Cu", "Zn"] to be consistent with other figures

            ax = fig.add_subplot(gs[i, 0])
            heat = sns.heatmap(
                subset_data,
                xticklabels=[],
                yticklabels=subset_data.index,
                ax=ax,
                center=0,
                vmin=-clip,
                vmax=clip,
                cmap="coolwarm",
                cbar=False,
                square=True,
            )

            ax.grid(False)
            ax.minorticks_off()
            ax.set_xlabel("")
            ax.set_ylabel("")

        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(subset_data.index)
        plt.savefig(f"{feature}_heatmap.pdf", bbox_inches='tight', facecolor='white', pad_inches=0.5)

        def plot_colorbar(clip=clip, cmap=cmap):
            fig = plt.figure(figsize=(10, 0.1))
            gs = gridspec.GridSpec(1, 3) 

            # Add colorbars
            ax_c = fig.add_subplot(gs[0, :3])
            norm_ads = plt.Normalize(vmin=-clip, vmax=clip)
            cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_ads, cmap="coolwarm"), cax=ax_c, orientation='horizontal')
            cb1.set_label(f"transformed feature")
            ax_c.minorticks_off()
            plt.savefig(f"{feature}_heatmap_cb.pdf", bbox_inches='tight', facecolor='white', pad_inches=0.5)
            return
        plot_colorbar()

        return plt.show()
    return (draw_3x3_response,)


@app.cell
def _(draw_3x3_response, plt):
    with plt.rc_context({'font.size': 25}):
        draw_3x3_response(80, "slope")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Principal Component Analysis (PCA)""")
    return


@app.cell
def _(PCA, feature_col_names, pd, transformed_combo_df):
    pcadata = transformed_combo_df[feature_col_names].copy()

    pca = PCA(n_components=2)
    latent_vectors = pca.fit_transform(pcadata)
    z1, z2 = pca.explained_variance_ratio_
    print(z1, z2)

    pcs = pd.DataFrame(data=latent_vectors, columns=['PC1', 'PC2'])
    pcs_and_exps = pd.concat([transformed_combo_df, pcs], axis=1)
    pcs_and_exps
    return pcs_and_exps, z1, z2


@app.cell
def _(Line2D, np, plt):
    def plot_PCA(pcs_and_exps, z1, z2):
        pc1 = pcs_and_exps['PC1']
        pc2 = pcs_and_exps['PC2']
        gas = pcs_and_exps['gas']
        ppm = pcs_and_exps['ppm']

        # create dictionary for gas and corresponding colors
        colordict = {'CO': 'grey', 'H2S': 'goldenrod', 'NO': 'teal'}

        fig, ax = plt.subplots(figsize=(10, 5))

        gas_types = [('CO','CO'),('H2S','H$_2$S'),('NO','NO')] # gas label for accessing data and gas label for legend
        ppm_values = pcs_and_exps['ppm'].unique()


        ppm_values.sort()

        # create the bubble plot and legend handles
        gas_legend_elements = []
        ppm_legend_elements = []
        for gas_type, gas_label in gas_types:
            gas_mask = (gas == gas_type)
            scatter = ax.scatter(pc1[gas_mask], pc2[gas_mask], s=(3 * ppm[gas_mask]),
                                edgecolors=colordict[gas_type], linewidths=1.5, facecolors='none', clip_on=False)
            gas_legend_elements.append(Line2D([0], [0], marker='o', color='w', label=gas_label,
                                            markeredgecolor=colordict[gas_type], markerfacecolor='none', markersize=15))

        ppm_legend_elements = [Line2D([0], [0], marker='o', color='w', label=str(ppm_value)+" ppm",
                                markerfacecolor='w', markeredgecolor='black', ms=2 * np.sqrt(ppm_value)) for ppm_value in ppm_values]

        # set x and y axis labels and limits
        ax.set_xlabel(f'PC1 score [{round(z1*100, 1)}%]')
        ax.set_ylabel(f'PC2 score\n[{round(z2*100, 1)}%]')
        ax.grid(False)

        # create the legends
        gas_legend = ax.legend(handles=gas_legend_elements, title=None, loc=(1, -.1), frameon=False)
        ppm_legend = ax.legend(handles=ppm_legend_elements, title=None, loc=(-0.16, -1.1),
                            ncol=len(ppm_values), frameon=False)

        ax.add_artist(gas_legend)
        #ax.add_artist(ppm_legend)

        ax.set_aspect('equal')
        ax.minorticks_off()
        plt.axhline(y=0, color='grey')
        plt.axvline(x=0, color='grey')
        plt.tight_layout()

        # Adjust the layout
        plt.savefig("PCA.png", bbox_extra_artists=(gas_legend, ppm_legend), bbox_inches='tight')
        return plt.show()

    return (plot_PCA,)


@app.cell
def _(pcs_and_exps, plot_PCA, plt, z1, z2):
    with plt.rc_context({'font.size': 22}):
        plot_PCA(pcs_and_exps, z1, z2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Supervised Learning""")
    return


@app.cell
def _(PowerTransformer, assemble_array_response, data, feature_col_names):
    train_transformer = PowerTransformer()

    train_combo = assemble_array_response(data, ppms=[10, 20, 80])
    X_train = train_transformer.fit_transform(train_combo[feature_col_names])

    test_combo = assemble_array_response(data, ppms=[40])
    X_test = train_transformer.transform(test_combo[feature_col_names])
    return X_test, X_train, test_combo, train_combo


@app.cell
def _(train_combo):
    train_combo
    return


@app.cell
def _(test_combo):
    test_combo
    return


@app.cell
def _(KNeighborsClassifier, X_train, np, train_combo):
    neighbors = int(np.sqrt(len(train_combo)))
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    knn.fit(X_train, train_combo.gas)
    return (knn,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    ListedColormap,
    X_test,
    knn,
    np,
    plt,
    test_combo,
):
    with plt.rc_context({'font.size': 25}):
        cmap = ListedColormap(["white", *plt.cm.Greens(np.linspace(0.1, 1))])
        disp = ConfusionMatrixDisplay.from_estimator(knn,
                                                     X_test,
                                                     test_combo.gas,
                                                     display_labels=gases_to_pretty_name(knn.classes_),
                                                     cmap=cmap)
        colorbar = disp.figure_.axes[-1] 
        colorbar.minorticks_off()  
        plt.minorticks_off()
        plt.tight_layout()
        plt.grid(False)
        plt.savefig("confusionmatrix.png")
        plt.show()
    return (cmap,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
