from stuff.data_and_data_processing.data_loading_wrapper import *


def get_stats(df, name='name not specified'):
    n_duplicates = len(df) - len(df.drop_duplicates())
    avg_expr_size = df["Equation"].str.split().str.len().mean()
    avg_n_nums = df["Numbers"].str.split().str.len().mean()
    print(f"printing data stats, dataset: {name}\nsize: {len(df)}\nduplicates: "
          f"{n_duplicates}\navg expr size: {avg_expr_size}\navg n nums: {avg_n_nums}")

def check_test_leak(df_train, df_test, name='name not specified'):
    df_train.drop_duplicates(inplace=True)
    #df_test.drop_duplicates(inplace=True)
    test0 = pd.merge(df_test, df_train, indicator=True, how='left', on=['Question','Numbers','Equation','Answer'])
    n_leaked = len(test0.loc[test0["_merge"] == "both"])
    print(f"printing test leakage, dataset: {name}\nleaked test data: {n_leaked} / {len(df_test)}")
    """
    testx = pd.merge(df_test,df_train, how='outer')
    test1 = pd.merge(df_test,df_train, indicator=True, how='outer').query('_merge=="left_only"')
    test2 = pd.merge(df_test,df_train, indicator=True, how='outer')
    combined = pd.concat([df_train, df_test])
    combined.drop_duplicates(inplace=True)
    """

    #print(f"test duplicates: {duplicates_in_test}")
    #print(f"test leak: {len(df_train) + len(df_test) - len(combined)} / {len(df_)}")

def check_griffiths_script():
    train = get_griffiths_all_train()
    train_m = get_griffiths_mawps_train()
    test = get_griffiths_mawps_test()
    get_stats(train, name='all train')
    get_stats(train_m, name='mawps train')
    get_stats(test, name='mawps test')
    check_test_leak(train, test, name="all")
    check_test_leak(train, train_m, name="sanity check")
    check_test_leak(train_m, test, name='m')

def check_stats_script():
    get_stats(get_mawps_mine_en(), name='mawps_mine_en')
    get_stats(get_mawps_mine_cs(), name='mawps_mine_cs')
    get_stats(get_mawps_or(),      name='mawps_or')
    get_stats(get_asdiva_mine_en(),name='asdiva_mine_en')
    get_stats(get_asdiva_mine_cs(),name='asdiva_mine_cs')
    get_stats(get_asdiva_or()     ,name='asdiva_or')
    get_stats(get_svamp_mine_en() ,name='svamp_mine_en')
    get_stats(get_svamp_mine_cs() ,name='svamp_mine_cs')
    #get_stats(get_svamp_or())
    get_stats(get_wp500cz()       ,name='wp500cz')

    check_test_leak(*get_mawps_mine_en(individual=True), name='mawps_mine_en')
    check_test_leak(*get_mawps_mine_cs(individual=True), name='mawps_mine_cs')
    check_test_leak(*get_mawps_or(individual=True),      name='mawps_or')
    check_test_leak(*get_asdiva_mine_en(individual=True),name='asdiva_mine_en')
    check_test_leak(*get_asdiva_mine_cs(individual=True),name='asdiva_mine_cs')
    check_test_leak(*get_asdiva_or(individual=True)     ,name='asdiva_or')
    check_test_leak(*get_svamp_mine_en(individual=True) ,name='svamp_mine_en')
    check_test_leak(*get_svamp_mine_cs(individual=True) ,name='svamp_mine_cs')
    #get_stats(get_svamp_or())

    get_stats(get_wp500cz()       ,name='wp500cz')

def check_math23k_script():
    get_stats(get_math23k_or(), name='math23k_or')
    get_stats(get_math23k_enb(), name='math23k_enb')
    get_stats(get_math23k_enh(), name='math23k_enh')
    get_stats(get_math23k_cz(),name='math23k_cz')

    check_test_leak(*get_math23k_or(individual=True), name='math23k_or')
    check_test_leak(*get_math23k_enb(individual=True), name='math23k_enb')
    check_test_leak(*get_math23k_enh(individual=True), name='math23k_enh')
    check_test_leak(*get_math23k_cz(individual=True), name='math23k_cz')

#check_griffiths_script()
#check_stats_script()
#check_math23k_script()
check_test_leak(pd.read_csv(f"train.csv"), pd.read_csv(f"dev.csv"), name='name not specified')