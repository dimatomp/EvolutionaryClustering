from sys import float_info

kmeans_algo = "KMeans"
affinity_algo = "Affinity_Propagation"
mean_shift_algo = "Mean_Shift"
ward_algo = "Ward"
dbscan_algo = "DBSCAN"
gm_algo = "Gaussian_Mixture"
bgm_algo = "Bayesian_Gaussian_Mixture"

num_algos = 7

# Metrica names can only consist of small english letters and dashes
purity_metric = "purity"
rand_index_metric = "rand-measure"
f_measure = "f-measure"
jaccard_index_metric = "jaccard-index"
dice_metric = "dice-index"


dunn_metric = "dunn"
cal_har_metric = "calinski-harabasz"
silhouette_metric = "silhouette"
davies_bouldin_metric = "db"
dunn31_metric = "gd31"
dunn41_metric = "gd41"
dunn51_metric = "gd51"
dunn33_metric = "gd33"
dunn43_metric = "gd43"
dunn53_metric = "gd53"
gamma_metric = "gamma"
cs_metric = "cs"
db_star_metric = "db-star"
sf_metric = "sf"
sym_metric = "sym"
cop_metric = "cop"
sv_metric = "sv"
os_metric = "os"
sym_bd_metric = "sym-db"
s_dbw_metric = "s-dbw"
c_ind_metric = "c-ind"

paused = "paused"
resume = "resume"
run = "run"

bandit_timeout = 30  # 5 # seconds for each bandit iteration
bandit_iterations = 40  # 10 # iterations number
batch_size = 40

tuner_timeout = bandit_timeout * (bandit_iterations + 1) / num_algos
smac_temp_dir = "/tmp/rm_me/"

n_samples = 500  # to generate data

tau = 0.5

proj_root = '~/WORK/MultiClustering/'
experiment_path = 'datasets/normalized/'
unified_data_path = 'datasets/unified/'

bad_cluster = float_info.max  # 100.0
in_reward = 1000.0
best_init = 1000000000.0

seeds = [1, 11, 111, 211, 311]

metrics = [
    davies_bouldin_metric,  # 1
    dunn_metric,  # 2
    cal_har_metric,  # 3, from scikit-learn
    silhouette_metric,  # 4, from scikit-learn
    dunn31_metric,  # 5
    dunn41_metric,  # 6
    dunn51_metric,  # 7
    dunn33_metric,  # 8
    dunn43_metric,  # 9
    dunn53_metric,  # 10
    # gamma_metric,  # 11 BROKEN
    cs_metric,  # 12
    db_star_metric,  # 13
    sf_metric,  # 14
    sym_metric,  # 15
    cop_metric,  # 16
    sv_metric,  # 17
    os_metric,  # 18
    s_dbw_metric,  # 19
    c_ind_metric  # 20
]

algos = [
    kmeans_algo,
    affinity_algo,
    mean_shift_algo,
    ward_algo,
    dbscan_algo,
    gm_algo,
    bgm_algo
]

noisy = False

algorithm = "rfrsls-ucb-SRSU-100"
