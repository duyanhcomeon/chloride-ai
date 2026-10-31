"""Training and evaluation on measured chloride *migration* (RCM) coefficient.

No conversion to apparent chloride diffusion D_app is performed. The RCM data
MUST NOT be directly fed into the Fick-based corrosion-initiation calculation.
Dataset provenance: Yang, Zhu, Wei (2025) https://doi.org/10.5061/dryad.66t1g1k6j

Method: 20% research-source-group holdout, 5-fold GroupKFold on training only.
Hyperparameters are predeclared, not tuned on the holdout. The best model is
chosen by grouped CV MAE; the holdout is checked ONCE for the chosen model.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

NUM_COLS=['fly_ash_pct','water_binder_ratio']
CAT_COLS=['cement_type','fly_ash_grade']
FEATURES=NUM_COLS+CAT_COLS
TARGET='d_rcm_28_m2_s'
GROUP='source_group'
AUTO='Tự động chọn bằng GroupKFold'
MODELS=(
 'Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest',
 'Extra Trees', 'Gradient Boosting', 'SVR (RBF)', 'KNN Regression', 'Neural Network (MLP)',
)


def _regressor(name):
 if name=='Linear Regression':return LinearRegression()
 if name=='Ridge Regression':return Ridge(alpha=8.0)
 if name=='Decision Tree':return DecisionTreeRegressor(max_depth=5,min_samples_leaf=3,random_state=42)
 if name=='Random Forest':return RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1)
 if name=='Extra Trees':return ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1)
 if name=='Gradient Boosting':return GradientBoostingRegressor(n_estimators=130,max_depth=2,learning_rate=0.05,min_samples_leaf=2,random_state=42)
 if name=='SVR (RBF)':return SVR(kernel='rbf',C=2.0,epsilon=.05)
 if name=='KNN Regression':return KNeighborsRegressor(n_neighbors=7,weights='distance')
 if name=='Neural Network (MLP)':return MLPRegressor(hidden_layer_sizes=(16,8),solver='lbfgs',alpha=.08,max_iter=350,random_state=42)
 raise ValueError('Thuật toán chưa hỗ trợ: '+str(name))


def _pipe(name):
 # Scaling after one-hot keeps each algorithm on a consistent feature matrix.
 pre=ColumnTransformer([
  ('numerical',StandardScaler(),NUM_COLS),
  ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT_COLS),
 ],sparse_threshold=0)
 return Pipeline([('prep',pre),('model',_regressor(name))])


def clean(df):
 required=FEATURES+[TARGET,GROUP]
 missing=[x for x in required if x not in df.columns]
 if missing:raise ValueError('File CSV thiếu cột: '+', '.join(missing))
 z=df.copy()
 for c in NUM_COLS+[TARGET]:z[c]=pd.to_numeric(z[c],errors='coerce')
 # Preserve original CSV blanks. Category 'khong_ghi' is used inside the ML pipeline only.
 for c in CAT_COLS+[GROUP]:
  z[c]=z[c].astype('string').str.strip()
 z['fly_ash_grade']=z['fly_ash_grade'].replace({'<NA>':pd.NA,'':pd.NA})
 z['cement_type']=z['cement_type'].replace({'':pd.NA})
 z[CAT_COLS]=z[CAT_COLS].fillna('khong_ghi').astype(str)
 z[GROUP]=z[GROUP].replace({'':pd.NA})
 ok=(z['fly_ash_pct'].between(0,100)&z['water_binder_ratio'].between(0.1,1.0)
     &z[TARGET].gt(0)&np.isfinite(z[TARGET])&z[GROUP].notna())
 z=z.loc[ok].copy().reset_index(drop=True)
 if len(z)<50:raise ValueError('Cần ít nhất 50 quan sát thực hợp lệ để chạy 9 thuật toán.')
 if z[GROUP].nunique()<10:raise ValueError('Cần ít nhất 10 nhóm nguồn bài báo để chia train/holdout an toàn.')
 return z


def _stats(real, predicted):
 real=np.asarray(real,float);predicted=np.asarray(predicted,float)
 return {'MAE_x1e12':float(mean_absolute_error(real,predicted)*1e12),
         'RMSE_x1e12':float(np.sqrt(mean_squared_error(real,predicted))*1e12),
         'R2':float(r2_score(real,predicted))}


def train_compare(df, selection=AUTO):
 """Run group-aware 5-fold CV on training, then evaluate selected on group holdout."""
 if selection!=AUTO and selection not in MODELS:raise ValueError('Mô hình không hợp lệ')
 data=clean(df)
 X=data[FEATURES].copy();y=np.log10(data[TARGET].to_numpy(float))
 groups=data[GROUP].to_numpy(str)
 train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
 Xtr=X.iloc[train_idx];Xte=X.iloc[test_idx]
 ytr=y[train_idx];yte=y[test_idx]
 gtr=groups[train_idx];gte=groups[test_idx]
 assert not(set(gtr)&set(gte))
 nfold=min(5,len(set(gtr)))
 cv=GroupKFold(n_splits=nfold)
 table=[]
 for name in MODELS:
  predictions=np.full(len(ytr),np.nan)
  for tr,va in cv.split(Xtr,ytr,gtr):
   estimator=_pipe(name)
   with warnings.catch_warnings():
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    estimator.fit(Xtr.iloc[tr],ytr[tr])
   predictions[va]=estimator.predict(Xtr.iloc[va])
  if not np.isfinite(predictions).all():raise ValueError('CV tạo giá trị không hữu hạn: '+name)
  metric=_stats(10**ytr,10**predictions)
  table.append({'Thuật toán':name,'CV MAE (×10⁻¹² m²/s)':metric['MAE_x1e12'],
                'CV RMSE (×10⁻¹² m²/s)':metric['RMSE_x1e12'],'CV R²':metric['R2'],'Số folds':nfold})
 leaderboard=pd.DataFrame(table).sort_values(['CV MAE (×10⁻¹² m²/s)','CV RMSE (×10⁻¹² m²/s)']).reset_index(drop=True)
 best=str(leaderboard.iloc[0]['Thuật toán'])
 chosen=best if selection==AUTO else selection
 estimator=_pipe(chosen)
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=ConvergenceWarning)
  estimator.fit(Xtr,ytr)
 pred_test=(10**estimator.predict(Xte))
 measured=10**yte
 metric=_stats(measured,pred_test)
 baseline=np.full(len(yte),np.median(10**ytr))
 holdout=pd.DataFrame({'D_RCM_28_thuc_te_x1e12':measured*1e12,
                      'D_RCM_28_du_bao_x1e12':pred_test*1e12,
                      'source_group':gte,
                      'excel_row':data.iloc[test_idx]['source_excel_row'].to_numpy() if 'source_excel_row' in data else test_idx})
 result={'selected':chosen,'best_cv':best,'train_count':len(train_idx),
         'test_count':len(test_idx),'train_source_count':len(set(gtr)),
         'test_source_count':len(set(gte)),'total_count':len(data),
         'table':leaderboard,'holdout':holdout,'holdout_scores':metric,
         'baseline_MAE_x1e12':_stats(measured,baseline)['MAE_x1e12'],
         'model':estimator,
         'training_rows':data.iloc[train_idx].copy().reset_index(drop=True),
         'bounds':{k:(float(Xtr[k].min()),float(Xtr[k].max())) for k in NUM_COLS},
         'cement_types':sorted(data['cement_type'].dropna().unique()),
         'fly_ash_grades':sorted(data['fly_ash_grade'].dropna().unique()),
         'citation':'Yang, Zhu, Wei (2025) https://doi.org/10.5061/dryad.66t1g1k6j'}
 return result


def predict_rcm(result,fly_ash_pct,water_binder_ratio,cement_type,fly_ash_grade):
 for col,value in [('fly_ash_pct',fly_ash_pct),('water_binder_ratio',water_binder_ratio)]:
  a,b=result['bounds'][col]
  if not a<=value<=b:raise ValueError(f'{col} = {value} ngoài miền dữ liệu train: {a:.3f}–{b:.3f}')
 if cement_type not in result['cement_types'] or fly_ash_grade not in result['fly_ash_grades']:
  raise ValueError('Loại xi măng hoặc cấp tro bay chưa có trong dữ liệu; không ngoại suy.')
 X=pd.DataFrame([{'fly_ash_pct':float(fly_ash_pct),
                  'water_binder_ratio':float(water_binder_ratio),
                  'cement_type':str(cement_type),
                  'fly_ash_grade':str(fly_ash_grade)}])
 return float(10**result['model'].predict(X)[0])
