import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from xgboost import XGBClassifier

from ParsianLoan import ParsianLoan

# اطلاعات اتصال به MySQL
DATABASE_URL = "mysql+pymysql://root:pass@localhost:3306/ln"

# ایجاد Engine
engine = create_engine(DATABASE_URL, echo=True)

# ایجاد Session
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()


def preProcessDataFromDB(session, limit_records=10000):
    """
    این متد داده‌ها را از دیتابیس می‌خواند، تعداد رکوردها را محدود می‌کند،
    مقدارهای `status` را به `0` و `1` تبدیل می‌کند، داده‌های گمشده را پر می‌کند،
    داده‌های اسمی را کدگذاری کرده و در نهایت مجموعه داده را آماده‌سازی می‌کند.
    """

    # دریافت تعداد محدود‌شده رکوردها از دیتابیس
    loans = session.query(ParsianLoan).limit(limit_records).all()
    df = pd.DataFrame([loan.__dict__ for loan in loans])
    df.drop(columns=["_sa_instance_state"], inplace=True)

    print(f"✅ {len(df)} رکورد از دیتابیس دریافت شد.")

    # انتخاب `status` به عنوان برچسب نکول
    label_column = 'status'

    if label_column not in df.columns:
        raise ValueError(f"ستون '{label_column}' در داده وجود ندارد. لطفاً نام صحیح ستون برچسب را مشخص کنید.")

    print(f"ستون برچسب انتخاب شده: {label_column}")

    # نمایش مقدارهای `status`
    print("مقدارهای `status` قبل از تبدیل:")
    print(df[label_column].value_counts())

    # تعریف وضعیت‌های نکول‌شده (`1`) و غیر نکول (`0`)
    default_statuses = ['مشكوك الوصول', 'معوق', 'سررسيد گذشته']

    df[label_column] = df[label_column].apply(lambda x: 1 if x in default_statuses else 0)

    # نمایش تعداد نمونه‌های نکول و غیرنکول
    print("تعداد برچسب‌های نکول و غیرنکول پس از تبدیل:")
    print(df[label_column].value_counts())

    # جدا کردن `X` و `y`
    X = df.drop([label_column], axis=1)
    y = df[label_column]

    # تقسیم داده‌ها به مجموعه‌ی آموزشی و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


# این تابع، کل پروسه پیش‌پردازش شامل پاکسازی مقادیر گمشده، حذف ویژگی‌های نامناسب و اعمال SMOTE را انجام می‌دهد
def preProcessData(filepath):
    # داده را بخوان
    df = pd.read_csv(filepath)
    # اگر ستون‌هایی به‌عنوان کلید یا غیرضروری داشتیم اینجا حذف می‌کنیم (در صورت نیاز)
    # اینجا هم می‌توان ویژگی‌هایی که مقاله پیشنهاد داده را نگه داشت یا حذف کرد
    # در صورت وجود ستون های اضافی اینجا drop می‌کنیم
    # به عنوان مثال:
    # df.drop(['Unwanted_Column'], axis=1, inplace=True)

    # داده‌های گمشده با میانگین یا هر روش دیگر پر شود
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # اگر داده اسمی داریم باید آن را LabelEncoder کنیم
    # در این مثال فرض می‌کنیم ستون خاصی به صورت اسمی باشد
    # می‌توان طبق نیاز همه ستون‌های اسمی را تبدیل کرد
    # مثال:
    # df_imputed['SomeCategoricalFeature'] = LabelEncoder().fit_transform(df_imputed['SomeCategoricalFeature'])

    # فرض می‌کنیم ستون 'label' برچسب پیش‌فرض (1) یا غیر پیش‌فرض (0) است
    X = df_imputed.drop(['label'], axis=1)
    y = df_imputed['label']

    # حذف ویژگی‌های نامناسب یا کم‌اهمیت طبق مقاله (اینجا فقط یک مثال ساده)
    # مثلاً اگر مقاله گفته ویژگی 'A2' حذف شود می‌نویسیم:
    # if 'A2' in X.columns:
    #     X.drop(['A2'], axis=1, inplace=True)

    # حالا داده را به دو بخش train , test برای ادامه کار تقسیم می‌کنیم
    # هرچند می‌توانیم این بخش را بیرون هم انجام دهیم
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # اعمال SMOTE برای متعادل‌سازی
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, y_train_res, X_test, y_test


# این تابع مدل LightGBM را با داده آموزشی آموزش می‌دهد و احتمال پیش‌فرض را خروجی می‌کند
def trainLightGBMModel(X_train, y_train, X_test):
    # یک شیء LGBM بساز
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    # مدل را آموزش بده
    lgbm.fit(X_train, y_train)
    # برای نمونه‌های تست احتمال پیش‌فرض خروجی بگیر
    p_pred = lgbm.predict_proba(X_test)[:, 1]
    return p_pred, lgbm


# محاسبه ضرر اشتباه بر اساس جریان نقدی
# تابع زیر ضررهای PN و NP را برای هر نمونه محاسبه می‌کند
# این ضررها را در دنیای واقعی با توجه به مبلغ و سود وام از دیتای جریان نقدی استخراج می‌کنیم
def computeLosses(cash_flow_info):
    # در اینجا cash_flow_info دیتافریمی است که حاوی اطلاعات اصل و سود وام ها برای هر رکورد است
    # مثلاً ستون 'principal' و 'interest' را دارد و برچسب پیش‌فرض هم دارد
    # خروجی ضررPN = interest
    # و ضررNP = principal + interest
    # برای سادگی فرض می‌کنیم خروجی را در آرایه های lambdaPN_arr و lambdaNP_arr برمی‌گردانیم
    principal = cash_flow_info['principal'].values
    interest = cash_flow_info['interest'].values
    # اینجا فرض می‌کنیم ترتیب cash_flow_info دقیقاً با X_test همراستا باشد و rowها یک‌به‌یک منطبق باشند
    # در عمل باید مراقب ایندکس‌ها بود

    lambdaPN_arr = []
    lambdaNP_arr = []
    for i in range(len(principal)):
        pn_val = interest[i]  # ضرر در صورتی که پیش‌فرض نباشد اما ما اشتباهاً پیش‌فرض کنیم (سود از دست رفته)
        np_val = principal[i] + interest[i]  # ضرر در صورتی که پیش‌فرض باشد اما ما اشتباهاً غیر پیش‌فرض کنیم
        lambdaPN_arr.append(pn_val)
        lambdaNP_arr.append(np_val)

    return np.array(lambdaPN_arr), np.array(lambdaNP_arr)


# این تابع ضررهای تأخیردار را با ضرایب u و v حساب می‌کند
# در اصل همان λ_BP = u * λ_NP و λ_BN = v * λ_PN
def delayedLosses(lambdaPN_arr, lambdaNP_arr, u, v):
    lambdaBP_arr = u * lambdaNP_arr
    lambdaBN_arr = v * lambdaPN_arr
    return lambdaBP_arr, lambdaBN_arr


# این تابع تابع هدف چندهدفه را حساب می‌کند تا در NSGA-II استفاده شود
# که هدفش کمینه کردن مجموع ضرر تصمیم و کوچک کردن اندازه ناحیه مرزی است
# اینجا صرفاً یک نمونه‌سازی ساده داریم و کل ایده را نمایان می‌کنیم
def computeObjective(u, v, p_pred, lambdaPN_arr, lambdaNP_arr):
    # با داشتن u, v برای هر نمونه آستانه‌های α و β را حساب می‌کنیم
    # formula = alpha_i = (PN_i - BN_i) / ((PN_i - BN_i) + (BP_i - PP_i)) 
    # ولی PP_i = 0 و BN_i = v*PN_i , BP_i = u*NP_i
    # همچنین beta_i = BN_i / (BN_i + (NP_i - BP_i)) که NP_i = λ_NP
    alpha_arr = []
    beta_arr = []
    # محاسبه هزینه ها
    cost_total = 0.0
    # مقدار اندازه ناحیه مرزی برای شمارش اختلاف آلفا و بتا
    boundary_sum = 0.0

    for i in range(len(p_pred)):
        pn = lambdaPN_arr[i]
        np_ = lambdaNP_arr[i]
        bp = u * np_
        bn = v * pn
        # alpha_i
        numerator_alpha = (pn - bn)
        denominator_alpha = (pn - bn) + (bp - 0)  # چون PP=0
        if denominator_alpha == 0:
            alpha_i = 1.0
        else:
            alpha_i = numerator_alpha / denominator_alpha
        # beta_i
        numerator_beta = bn
        denominator_beta = bn + (np_ - bp)
        if denominator_beta == 0:
            beta_i = 0.0
        else:
            beta_i = numerator_beta / denominator_beta

        # اگر alpha_i < beta_i بشه ، طبق مقاله ممکنه نمونه تبدیل به دوطرفه شه
        # ولی ما باید تضمین کنیم u+v <= 1 که معمولا تضمین می‌کند alpha >= beta
        alpha_arr.append(alpha_i)
        beta_arr.append(beta_i)

        # حالا بر اساس p_pred[i] تصمیم می‌گیریم هزینه کدام
        p_val = p_pred[i]
        if p_val >= alpha_i:
            # اقدام پذیرش=> ضررPN * (1 - p) 
            cost_local = pn * (1 - p_val)
        elif p_val <= beta_i:
            # اقدام رد=> ضررNP * p
            cost_local = np_ * p_val
        else:
            # اقدام مرزی=> BP * p + BN * (1 - p)
            cost_local = (bp * p_val) + (bn * (1 - p_val))

        cost_total += cost_local
        boundary_sum += (alpha_i - beta_i)  # اندازه ناحیه مرزی

    return cost_total, boundary_sum


# پیاده سازی ساده NSGA-II برای یافتن u و v
# برای اختصار جزئیات کامل الگوریتم ژنتیک حذف شده و فقط ساختار پایه داریم
def nsga2_find_uv(p_pred, lambdaPN_arr, lambdaNP_arr, population_size=30, generations=20):
    # ایجاد جمعیت اولیه
    population = []
    for _ in range(population_size):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        # اطمینان از اینکه u+v <=1
        if u + v > 1:
            # ساده ترین حالت اینکه یا v را کم کنیم
            v = 1 - u
        population.append((u, v))

    for gen in range(generations):
        # براساس computeObjective، هزینه را حساب می‌کنیم
        # مرتب‌سازی غیرمغلوب را انجام می‌دهیم (اینجا به صورت خیلی ساده فقط مرتب می‌کنیم)
        # تکثیر و تقاطع و جهش... در نمونه واقعی باید انجام گیرد
        new_pop = []
        for (u, v) in population:
            cost_val, boundary_val = computeObjective(u, v, p_pred, lambdaPN_arr, lambdaNP_arr)
            new_pop.append((u, v, cost_val, boundary_val))
        # مرتب‌سازی ساده با اولویت cost_val و بعد boundary_val
        new_pop.sort(key=lambda x: (x[2], x[3]))
        # بقا
        top_half = new_pop[:population_size // 2]
        # بازترکیب نسل بعد ساده (اینجا خیلی مختصر)
        new_population = []
        for i in range(len(top_half)):
            u1, v1, c1, bd1 = top_half[i]
            # جهش ساده
            u1 += random.uniform(-0.01, 0.01)
            v1 += random.uniform(-0.01, 0.01)
            if u1 < 0: u1 = 0
            if v1 < 0: v1 = 0
            if u1 + v1 > 1:
                v1 = 1 - u1
            new_population.append((u1, v1))
        population = new_population
        # و اگر تعداد جمعیت کم است پرش می‌کنیم
        while len(population) < population_size:
            # تولید تصادفی
            utmp = random.uniform(0, 1)
            vtmp = random.uniform(0, 1)
            if utmp + vtmp > 1:
                vtmp = 1 - utmp
            population.append((utmp, vtmp))

    # در پایان بهترین راهکار را برمی‌گردانیم
    best_val = 1e15
    best_uv = (0, 0)
    best_bound = 1e15
    for (u, v) in population:
        cost_val, boundary_val = computeObjective(u, v, p_pred, lambdaPN_arr, lambdaNP_arr)
        if cost_val < best_val:
            best_val = cost_val
            best_bound = boundary_val
            best_uv = (u, v)
        elif abs(cost_val - best_val) < 1e-9 and boundary_val < best_bound:
            best_uv = (u, v)
            best_bound = boundary_val
    return best_uv


# تابع برای اعمال تصمیم سه‌طرفه با آستانه‌های α و β بر هر نمونه
# برچسب نهایی را بازمی‌گرداند (POS=1 یعنی پیش‌فرض، NEG=0 یعنی غیرپیش‌فرض، BND=-1 یعنی مرزی)
def applyThreeWayDecision(p_pred, lambdaPN_arr, lambdaNP_arr, u, v):
    labels = []
    boundary_indices = []
    for i in range(len(p_pred)):
        p_val = p_pred[i]
        pn = lambdaPN_arr[i]
        np_ = lambdaNP_arr[i]
        bp = u * np_
        bn = v * pn
        # alpha_i
        numerator_alpha = (pn - bn)
        denominator_alpha = (pn - bn) + (bp - 0)
        if denominator_alpha == 0:
            alpha_i = 1.0
        else:
            alpha_i = numerator_alpha / denominator_alpha
        # beta_i
        numerator_beta = bn
        denominator_beta = bn + (np_ - bp)
        if denominator_beta == 0:
            beta_i = 0.0
        else:
            beta_i = numerator_beta / denominator_beta

        if p_val >= alpha_i:
            # POS
            labels.append(1)
        elif p_val <= beta_i:
            # NEG
            labels.append(0)
        else:
            # BND
            labels.append(-1)
            boundary_indices.append(i)
    return np.array(labels), boundary_indices


# مدل استکینگ را آموزش می‌دهد
# اینجا برای مثال، لایه اول را چند الگوریتم تعریف می‌کنیم و خروجی‌ پیش‌بینی آنها وارد متا کلاسفایر می‌شود
def trainStacking(X_train, y_train, base_models, meta_model):
    # اینجا از KFold برای تولید خروجی base_models روی داده آموزش بهره می‌بریم
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    first_level_predictions = np.zeros((len(X_train), len(base_models)))  # خروجی هر مدل پایه برای هر نمونه
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    for idx, model in enumerate(base_models):
        fold_i = 0
        for train_index, val_index in kf.split(X_train_np):
            X_tr, X_val = X_train_np[train_index], X_train_np[val_index]
            y_tr, y_val = y_train_np[train_index], y_train_np[val_index]
            clf = model
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_val)[:, 1]
            first_level_predictions[val_index, idx] = preds
            fold_i += 1

    meta_model.fit(first_level_predictions, y_train_np)
    return base_models, meta_model


# پیش‌بینی استکینگ روی نمونه‌های جدید
def predictStacking(X_test, base_models, meta_model):
    base_preds = np.zeros((len(X_test), len(base_models)))
    for idx, model in enumerate(base_models):
        p = model.predict_proba(X_test)[:, 1]
        base_preds[:, idx] = p
    final_preds = meta_model.predict(base_preds)
    return final_preds


# کد اصلی که مراحل مقاله را اجرا می‌کند
if __name__ == "__main__":
    # مرحله اول: پیش‌پردازش داده
    # X_train_res, y_train_res, X_test, y_test = preProcessData('data.csv')
    X_train_res, y_train_res, X_test, y_test = preProcessDataFromDB(session)

    # استفاده از اطلاعات جریان نقدی واقعی موجود در X_test (فرض می‌کنیم ستون‌های 'principal' و 'interest' در X_test موجود هستند)
    data_test_cashflow = X_test[['approval_amount', 'interest_amount']]

    # مرحله دوم: آموزش مدل LGBM و محاسبه احتمال پیش‌فرض
    p_pred_test, lgbm_model = trainLightGBMModel(X_train_res, y_train_res, X_test)

    # ضرر PN: که برابر با سود (interest) هست؛ یعنی اگر ما به اشتباه وامی رو به عنوان غیر نکول در نظر بگیریم، سود از دست رفته چقدر میشه.
    # ضرر NP: که برابر با مجموع اصل و سود (principal + interest) هست؛ یعنی اگر ما به اشتباه وامی رو به عنوان نکول در نظر بگیریم، کل مبلغ ضرر از دست رفته چقدر میشه.

    # مرحله سوم: محاسبه ضررهای PN و NP بر اساس جریان نقدی
    lambdaPN_arr_test, lambdaNP_arr_test = computeLosses(data_test_cashflow)

    # مرحله چهارم: با توجه به تابع چندهدفه از NSGA-II برای یافتن u,v بهینه کمک می‌گیریم
    best_u, best_v = nsga2_find_uv(p_pred_test, lambdaPN_arr_test, lambdaNP_arr_test,
                                   population_size=20, generations=10)

    # مرحله پنجم: اعمال آستانه های سه‌طرفه برای تست
    twd_labels, boundary_indices = applyThreeWayDecision(p_pred_test, lambdaPN_arr_test, lambdaNP_arr_test,
                                                         best_u, best_v)

    # نمونه‌های مرزی را جدا می‌کنیم
    X_test_bnd = X_test.iloc[boundary_indices]
    y_test_bnd = y_test.iloc[boundary_indices]

    # مرحله ششم: مدل استکینگ را برای تصمیم‌گیری نهایی بر نمونه‌های مرزی آماده می‌کنیم
    base_models = [
        LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=0),
        RandomForestClassifier(n_estimators=100, random_state=0),
        XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=0),
        GradientBoostingClassifier(n_estimators=100, random_state=0),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        AdaBoostClassifier(n_estimators=100, random_state=0),
    ]
    meta_model = LogisticRegression()

    # برای آموزش استکینگ نیاز داریم از داده‌های آموزشی استفاده کنیم
    # (در مقاله آمده که لایه اول و دوم با مجموعه آموزشی train می‌شوند)
    # ولی اینجا برای نمونه ساده همان را استفاده می‌کنیم
    base_models_trained, meta_model_trained = trainStacking(X_train_res, y_train_res, base_models, meta_model)

    # حالا روی نمونه‌های مرزی در تست خروجی می‌دهیم
    y_pred_bnd = predictStacking(X_test_bnd, base_models_trained, meta_model_trained)

    # جایگزین برچسب نهایی در آرایه twd_labels برای نمونه‌های مرزی
    for i, idx in enumerate(boundary_indices):
        twd_labels[idx] = y_pred_bnd[i]

    # حال که برچسب سه‌طرفه (یا استکینگ برای مرزی‌ها) تعیین شد، عملکرد را ارزیابی می‌کنیم
    # محاسبه ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_test, twd_labels)
    # cm[0,0] = TN , cm[0,1] = FP , cm[1,0] = FN , cm[1,1] = TP
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # دقت متعادل
    # اگر تقسیم بر صفر نشود
    if (TP + FP) == 0 or (TN + FN) == 0:
        balanced_acc = 0
    else:
        balanced_acc = ((TP / (TP + FP)) + (TN / (TN + FN))) / 2.0

    # محاسبه AUC
    auc_score = roc_auc_score(y_test, twd_labels)

    # محاسبه F-measure و G-mean
    precision = 0
    recall = 0
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        fmeasure = 0
    else:
        fmeasure = 2 * (precision * recall) / (precision + recall)
    gmean = 0
    if (TP + FN) != 0 and (TN + FP) != 0:
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        gmean = (sensitivity * specificity) ** 0.5

    # محاسبه هزینه تصمیم
    # اگر پیش‌فرض است (y=1) ولی اشتباه پیش‌بینی شده => cost = λ_NP
    # اگر غیرپیش‌فرض است (y=0) ولی اشتباه پیش‌بینی شده => cost = λ_PN
    # در اینجا باید هر سطر تست را بررسی کنیم
    total_cost = 0.0
    y_test_np = np.array(y_test)
    twd_labels_np = np.array(twd_labels)
    for i in range(len(y_test_np)):
        if y_test_np[i] == 1 and twd_labels_np[i] == 0:
            # پیش‌فرض ولی ما گفتیم غیرپیش‌فرض
            total_cost += lambdaNP_arr_test[i]
        elif y_test_np[i] == 0 and twd_labels_np[i] == 1:
            # غیرپیش‌فرض ولی ما گفتیم پیش‌فرض
            total_cost += lambdaPN_arr_test[i]

    # نمایش نتایج
    print("Balanced Accuracy:", balanced_acc)
    print("AUC:", auc_score)
    print("F-Measure:", fmeasure)
    print("G-Mean:", gmean)
    print("Decision Cost:", total_cost)
    print("Confusion Matrix (TN, FP, FN, TP):", TN, FP, FN, TP)

# پایان کد
