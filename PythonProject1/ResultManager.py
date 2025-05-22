# result_manager.py
import os

import matplotlib.pyplot as plt
from matplotlib import cm


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

    def plot_rfecv(self , cv_scores):
        feature_counts = range(1, len(cv_scores) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(feature_counts, cv_scores, marker='o', linestyle='-')
        plt.title("RFECV نتایج انتخاب ویژگی")
        plt.xlabel("تعداد ویژگی‌های منتخب")
        plt.ylabel("امتیاز CV (ROC AUC)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rfecv_plot.png", dpi=300)  # ذخیره به عنوان تصویر
        plt.show()