# result_manager.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
import seaborn as sns


class ResultManager:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_csv(self, df, filename):
        """ذخیرهٔ DataFrame به صورت CSV"""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f'💾 CSV saved → {path}')

    def save_pareto_plot(self, res, fold):
        """ذخیرهٔ نمودار جبههٔ پارتو"""
        f1, f2 = res.F[:, 0], res.F[:, 1]
        plt.figure(figsize=(6, 4))
        plt.scatter(f2, f1, c='steelblue', s=25, alpha=0.8, edgecolor='k')
        plt.gca().invert_xaxis()
        plt.xlabel('f₂  (Border Width ∑(α-β))')
        plt.ylabel('f₁  (Decision Cost)')
        plt.title(f'Pareto Front – Fold {fold}')
        plt.tight_layout()
        fname = os.path.join(self.output_dir, f'pareto_fold{fold}.png')
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f'📈 Pareto front saved → {fname}')

    def plot_sensitivity(self, sens_df):
        """ذخیرهٔ نمودارهای تحلیل حساسیت"""
        # نمودار هزینه تصمیم
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        norm = plt.Normalize(sens_df['NGen'].min(), sens_df['NGen'].max())
        scatter = ax1.scatter(sens_df['PopSize'], sens_df['DecisionCost'],
                              s=sens_df['NumBND'] * 6, c=sens_df['NGen'],
                              cmap=cm.viridis, norm=norm, alpha=0.85,
                              edgecolor='k', linewidth=0.6)
        ax1.set_xlabel('Population Size')
        ax1.set_ylabel('Decision Cost (f₁)')
        ax1.set_title('NSGA-II Sensitivity: Cost vs Population Size')
        for _, r in sens_df.iterrows():
            ax1.text(r.PopSize, r.DecisionCost, f"BND={r.NumBND}", fontsize=8)
        cbar = fig1.colorbar(scatter, ax=ax1)
        cbar.set_label('Number of Generations')
        fig1.tight_layout()
        path1 = os.path.join(self.output_dir, 'nsga_sensitivity_cost.png')
        fig1.savefig(path1, dpi=300)

        # نمودار زمان اجرا
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(sens_df['PopSize'], sens_df['Seconds'],
                 marker='o', linestyle='-', color='#1f77b4')
        for _, r in sens_df.iterrows():
            ax2.text(r.PopSize, r.Seconds, f"{r.Seconds:.1f}s", fontsize=8)
        ax2.set_xlabel('Population Size')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('NSGA-II Sensitivity: Runtime vs Population Size')
        path2 = os.path.join(self.output_dir, 'nsga_sensitivity_runtime.png')
        fig2.tight_layout()
        fig2.savefig(path2, dpi=300)

        print(f'📊 Sensitivity plots saved → {path1} / {path2}')


    def plot_rfecv_feature_importance(self,rfecv, feature_names, top_n=20):
        """
        رسم نمودار افقی اهمیت ویژگی‌ها پس از اجرای RFECV
        rfecv: شیء آموزش‌دیده RFECV
        feature_names: نام تمام ویژگی‌های اولیه (لیستی یا آرایه‌ای)
        top_n: تعداد ویژگی‌های برتر برای نمایش
        """
        # دریافت نام و اهمیت ویژگی‌های انتخاب‌شده
        selected_mask = rfecv.support_
        selected_feats = np.array(feature_names)[selected_mask]
        importances = rfecv.estimator_.feature_importances_

        # ساخت DataFrame و مرتب‌سازی
        df_imp = pd.DataFrame({
            'feature': selected_feats,
            'importance': importances
        })
        df_imp = df_imp.sort_values('importance', ascending=True).tail(top_n)

        # رسم نمودار افقی
        plt.figure(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_imp)))
        plt.barh(df_imp['feature'], df_imp['importance'], color=colors)
        plt.title('feature importance', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig('rfecv_feature_importance.png', dpi=300)
        plt.show()

    def plot_label_count_before_smote(self, label_counts: pd.Series) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        plt.figure(figsize=(10, 6))

        # Ensure the index is numeric
        label_counts.index = label_counts.index.astype(int)

        # Convert to DataFrame with English column names
        label_df = pd.DataFrame({'Label': label_counts.index, 'Frequency': label_counts.values})

        sns.barplot(
            x='Label',
            y='Frequency',
            data=label_df,
            hue='Label',
            dodge=False,
            palette=['#4CAF50', '#FF6F61'],
            legend=False
        )

        plt.title('Label Distribution Before SMOTE', fontsize=18)
        plt.xlabel('Labels (0: Non-Default, 1: Default)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(ticks=[0, 1], labels=['Non-Default', 'Default'])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_prob_distribution(self, y_true, proba, fold):

        # Create DataFrame for easy plotting
        df = pd.DataFrame({'Default Probability': proba, 'True Label': y_true})
        plt.figure(figsize=(8, 5))

        # Density plot for Non-Default class
        sns.histplot(df[df['True Label'] == 0]['Default Probability'],
                     color='#4CAF50', label='Non-Default', kde=True, stat='density', bins=20, alpha=0.6)

        # Density plot for Default class
        sns.histplot(df[df['True Label'] == 1]['Default Probability'],
                     color='#FF6F61', label='Default', kde=True, stat='density', bins=20, alpha=0.6)

        plt.title(f'Default Probability Distribution – Fold {fold}', fontsize=14)
        plt.xlabel('Predicted Default Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title='Class', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'prob_dist_fold_{fold}.png', dpi=300)  # Save figure
        plt.show()

    def build_and_save_loss_matrix(self,lam_np, lam_pn, u, v,
                                   sample_idx, fold,
                                   output_dir='results'):
        """
        ماتریس زیان نمونه‌ی مشخص را می‌سازد و در فایل CSV ذخیره می‌کند.

        پارامترها:
          lam_np      λ_NP برای آن نمونه
          lam_pn      λ_PN برای آن نمونه
          u, v        پارامترهای بهینه‌شده‌ی تصمیم تأخیری
          sample_idx  اندیس (index) نمونه در X_te
          fold        شماره‌ی فولد
          output_dir  پوشه‌ای که فایل در آن ذخیره می‌شود
        """
        # اطمینان از وجود پوشه
        os.makedirs(output_dir, exist_ok=True)

        # تعریف اقدامات و ستون‌ها
        actions = ['پذیرش', 'تصمیم تأخیری', 'رد']
        cols = ['نکول', 'عدم نکول']
        mat = pd.DataFrame(index=actions, columns=cols, dtype=float)

        # پرکردن مقادیر ماتریس طبق جدول ۲
        mat.loc['پذیرش', 'نکول'] = 0
        mat.loc['پذیرش', 'عدم نکول'] = lam_pn
        mat.loc['رد', 'نکول'] = lam_np
        mat.loc['رد', 'عدم نکول'] = 0
        mat.loc['تصمیم تأخیری', 'نکول'] = u * lam_np
        mat.loc['تصمیم تأخیری', 'عدم نکول'] = v * lam_pn

        # آماده‌سازی برای ذخیره‌سازی
        mat_to_save = mat.reset_index().rename(columns={'index': 'اقدام'})
        mat_to_save.insert(0, 'نمونه', sample_idx)
        filename = f'loss_matrix_fold_{fold}_sample_{sample_idx}.csv'
        filepath = os.path.join(output_dir, filename)

        # ذخیره در CSV
        mat_to_save.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f'✅ ماتریس زیان نمونه {sample_idx} در فولد {fold} ذخیره شد: {filepath}')

        return mat

