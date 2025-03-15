import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class ObjectiveProblem(Problem):
    """
    مسئله بهینه‌سازی چندهدفه برای تعیین آستانه‌های سه‌راه با استفاده از NSGA-II.

    ورودی‌ها:
        - predicted_probs: آرایه‌ای از احتمال‌های پیش‌بینی شده (مثلاً احتمال نکول)
        - loss_PN: آرایه هزینه‌های مربوط به تصمیم اشتباه "قبول" برای نمونه‌های غیرنکول
        - loss_NP: آرایه هزینه‌های مربوط به تصمیم اشتباه "رد" برای نمونه‌های نکول
    """

    def __init__(self, predicted_probs, loss_PN, loss_NP):
        self.predicted_probs = predicted_probs
        self.loss_PN = loss_PN  # هزینه مربوط به تصمیم "قبول" (برای نمونه‌های غیرنکول)
        self.loss_NP = loss_NP  # هزینه مربوط به تصمیم "رد" (برای نمونه‌های نکول)

        # تعریف مسئله:
        # 2 متغیر (u, v)، 2 تابع هدف (f1: هزینه کل تصمیم، f2: اندازه منطقه مرزی) و
        # 1 محدودیت (u + v <= 1)
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([0.0, 0.0]),  # پایین‌ترین مقدار برای u و v
                         xu=np.array([1.0, 1.0]))  # بالاترین مقدار برای u و v

    def _evaluate(self, x, out, *args, **kwargs):
        """
        محاسبه تابع هدف برای هر راه‌حل (هر فرد).

        ورودی:
            - x: آرایه‌ای با ابعاد (تعداد افراد, 2) شامل مقادیر u و v برای هر فرد
        خروجی:
            - out["F"]: ماتریس اهداف؛ ستون اول هزینه کل تصمیم و ستون دوم اندازه منطقه مرزی
            - out["G"]: محدودیت (u + v - 1 <= 0)
        """
        n_individuals = x.shape[0]
        total_cost = np.zeros(n_individuals)  # f1: هزینه کلی تصمیم‌گیری
        boundary_size = np.zeros(n_individuals)  # f2: اندازه منطقه مرزی

        for i in range(n_individuals):
            u, v = x[i]
            # محاسبه تغییرات هزینه به کمک پارامترهای u و v:
            bp = u * self.loss_NP
            bn = v * self.loss_PN

            # محاسبه آستانه مثبت (alpha):
            numerator_alpha = self.loss_PN - bn
            denom_alpha = numerator_alpha + bp
            alpha = np.where(denom_alpha == 0, 1.0, numerator_alpha / denom_alpha)

            # محاسبه آستانه منفی (beta):
            numerator_beta = bn
            denom_beta = bn + (self.loss_NP - bp)
            beta = np.where(denom_beta == 0, 0.0, numerator_beta / denom_beta)

            # محاسبه هزینه محلی (local cost) برای هر نمونه:
            # - اگر احتمال پیش‌بینی (predicted_prob) بالاتر یا مساوی آستانه مثبت (alpha) باشد:
            #     هزینه = loss_PN * (1 - predicted_prob)
            # - اگر احتمال پیش‌بینی کمتر یا مساوی آستانه منفی (beta) باشد:
            #     هزینه = loss_NP * predicted_prob
            # - در غیر این صورت، هزینه ترکیبی از bp و bn استفاده می‌شود.
            local_cost = np.where(self.predicted_probs >= alpha,
                                  self.loss_PN * (1 - self.predicted_probs),
                                  np.where(self.predicted_probs <= beta,
                                           self.loss_NP * self.predicted_probs,
                                           bp * self.predicted_probs + bn * (1 - self.predicted_probs)))
            total_cost[i] = np.sum(local_cost)
            boundary_size[i] = np.sum(alpha - beta)

        # محدودیت: u + v باید کمتر یا مساوی 1 باشد، به این صورت که
        # g = u + v - 1 <= 0
        constraint = x[:, 0] + x[:, 1] - 1.0
        out["F"] = np.column_stack([total_cost, boundary_size])
        out["G"] = constraint.reshape(-1, 1)


def nsga2_find_uv(predicted_probs, loss_PN, loss_NP, pop_size=20, generations=10):
    """
    بهینه‌سازی پارامترهای u و v با استفاده از NSGA-II.

    ورودی‌ها:
        - predicted_probs: آرایه احتمال‌های پیش‌بینی شده
        - loss_PN: آرایه هزینه‌های تصمیم "قبول" (برای نمونه‌های غیرنکول)
        - loss_NP: آرایه هزینه‌های تصمیم "رد" (برای نمونه‌های نکول)
        - pop_size: اندازه جمعیت اولیه
        - generations: تعداد نسل‌های الگوریتم
    خروجی:
        - بهترین مقدار u و v به عنوان پارامترهای بهینه
    """
    problem = ObjectiveProblem(predicted_probs, loss_PN, loss_NP)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ('n_gen', generations), seed=1, verbose=False)

    # انتخاب بهترین فرد بر اساس مرتب‌سازی lexicographic:
    # ابتدا بر اساس f1 و سپس بر اساس f2 مرتب می‌شود.
    f = res.F
    idx = np.lexsort((f[:, 1], f[:, 0]))[0]
    best_u, best_v = res.X[idx]
    return best_u, best_v


# مثال استفاده:
if __name__ == "__main__":
    # فرض کنید این‌ها مقادیر نمونه‌ای هستند
    predicted_probs = np.array([0.1, 0.5, 0.8])
    loss_PN = np.array([10, 20, 30])
    loss_NP = np.array([5, 15, 25])

    best_u, best_v = nsga2_find_uv(predicted_probs, loss_PN, loss_NP, pop_size=20, generations=10)
    print("بهترین u:", best_u)
    print("بهترین v:", best_v)
