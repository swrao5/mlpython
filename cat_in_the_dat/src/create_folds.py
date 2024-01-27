# create_folds.py
# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
	df = pd.read_csv(r"C:\Users\raosw\projects\mlpython\cat_in_the_dat\input\train.csv")
	df = df.sample(frac=1).reset_index(drop=True)
	y = df.target.values
	# initiate k-fold stratified CV
	kfold = model_selection.StratifiedKFold(n_splits=5)
	
	# filling new column kfold
	for f, (t_, v_) in enumerate(kfold.split(X=df,y=y)):
		df.loc[v_, 'kfold'] = f
	df.to_csv(r"C:\Users\raosw\projects\mlpython\cat_in_the_dat\input\cat_train_folds.csv", index=False)