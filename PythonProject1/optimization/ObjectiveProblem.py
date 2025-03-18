import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class ThresholdOptimizationProblem(Problem):
    """
    مسئله تعیین مقیاس‌های تنظیمی برای محاسبه آستانه‌های تصمیم‌گیری.
    ورودی‌ها:
      - predicted_probs: آرایه احتمال‌های پیش‌بینی شده (برای هر نمونه)
      - false_pos_cost: آرایه هزینه‌های اشتباه «قبول» (False Positive)
      - false_neg_cost: آرایه هزینه‌های اشتباه «رد» (False Negative)
    """

    def __init__(self, predicted_probs, false_pos_cost, false_neg_cost):
        self.predicted_probs = predicted_probs
        self.false_pos_cost = false_pos_cost
        self.false_neg_cost = false_neg_cost
        # متغیرهای تصمیم:
        #   scale_fn: مقیاس تنظیمی برای هزینه false negative
        #   scale_fp: مقیاس تنظیمی برای هزینه false positive
        # تابع‌های هدف:
        #   objective1: مجموع هزینه‌های نمونه‌ها
        #   objective2: مجموع اختلاف آستانه‌های بالا و پایین (عرض مرز)
        # محدودیت:
        #   scale_fn + scale_fp <= 1
        super().__init__(
            # تعداد متغیرهای تصمیم: دو متغیر وجود دارد که همان scale_fn و scale_fp هستند.
            # population 2 ta soton date az in miyad
            n_var=2,
            # تعداد توابع هدف: دو تابع هدف برای بهینه‌سازی تعریف شده است.
            # الگوریتم سعی می‌کند مجموع هزینه‌ها را کم کند و عرض مرزهای تصمیم‌گیری را حداقل کند تا تصمیم‌گیری بهینه شود.
            n_obj=2,
            # تعداد محدودیت‌ها: تنها یک محدودیت (مجموع scale_fn و scale_fp باید کمتر یا مساوی ۱ باشد).
            n_constr=1,
            # دامنه پایین متغیرهای تصمیم: حداقل مقادیر مجاز برای scale_fn و scale_fp.
            xl=np.array([0.0, 0.0]),  # حداقل مقیاس‌ها
            # دامنه بالا متغیرهای تصمیم: حداکثر مقادیر مجاز برای scale_fn و scale_fp.
            xu=np.array([1.0, 1.0])
        )

    def calculate_adjusted_costs(self, scale_fn, scale_fp):
        # محاسبه هزینه‌های تعدیلی بر اساس مقیاس‌های ورودی
        adjusted_fn_cost = scale_fn * self.false_neg_cost   # هزینه false negative تعدیل‌شده
        adjusted_fp_cost = scale_fp * self.false_pos_cost     # هزینه false positive تعدیل‌شده
        return adjusted_fn_cost, adjusted_fp_cost

    def calculate_thresholds(self, adjusted_fn_cost, adjusted_fp_cost):
        # محاسبه آستانه بالا (برای تصمیم «قبول») و آستانه پایین (برای تصمیم «رد»)
        numerator_upper = self.false_pos_cost - adjusted_fp_cost
        denominator_upper = numerator_upper + adjusted_fn_cost
        upper_threshold = np.where(denominator_upper == 0, 1.0, numerator_upper / denominator_upper)

        numerator_lower = adjusted_fp_cost
        denominator_lower = adjusted_fp_cost + (self.false_neg_cost - adjusted_fn_cost)
        lower_threshold = np.where(denominator_lower == 0, 0.0, numerator_lower / denominator_lower)
        return upper_threshold, lower_threshold

    def compute_sample_costs(self, upper_threshold, lower_threshold, adjusted_fn_cost, adjusted_fp_cost):
        # برای هر نمونه:
        #   اگر احتمال پیش‌بینی >= آستانه بالا: هزینه = false_pos_cost * (1 - پیش‌بینی)
        #   اگر احتمال پیش‌بینی <= آستانه پایین: هزینه = false_neg_cost * پیش‌بینی
        #   در غیر این صورت: هزینه = adjusted_fn_cost * پیش‌بینی + adjusted_fp_cost * (1 - پیش‌بینی)
        sample_costs = np.where(self.predicted_probs >= upper_threshold,
                                self.false_pos_cost * (1 - self.predicted_probs),
                                np.where(self.predicted_probs <= lower_threshold,
                                         self.false_neg_cost * self.predicted_probs,
                                         adjusted_fn_cost * self.predicted_probs + adjusted_fp_cost * (1 - self.predicted_probs)))
        return sample_costs

    def _evaluate(self, population, out, *args, **kwargs):
        # تعداد راه‌حل‌های موجود در جمعیت
        # تعداد ردیف های یک ماتریس
        num_solutions = population.shape[0]
        # آرایه برای ذخیره مجموع هزینه هر راه‌حل
        total_costs = np.zeros(num_solutions)
        # آرایه برای ذخیره مجموع اختلاف آستانه‌ها (عرض مرز) هر راه‌حل
        total_boundary_width = np.zeros(num_solutions)

        for i in range(num_solutions):
            # استخراج مقیاس‌های تنظیمی از راه‌حل i
            scale_fn, scale_fp = population[i]
            # محاسبه هزینه‌های تعدیلی
            adj_fn_cost, adj_fp_cost = self.calculate_adjusted_costs(scale_fn, scale_fp)
            # محاسبه آستانه‌های بالا و پایین
            upper_threshold, lower_threshold = self.calculate_thresholds(adj_fn_cost, adj_fp_cost)
            # محاسبه هزینه‌های هر نمونه
            sample_costs = self.compute_sample_costs(upper_threshold, lower_threshold, adj_fn_cost, adj_fp_cost)
            # محاسبه مجموع هزینه‌ها و عرض مرز برای راه‌حل i
            total_costs[i] = np.sum(sample_costs)
            total_boundary_width[i] = np.sum(upper_threshold - lower_threshold)

        # محدودیت: مجموع مقیاس‌های تنظیمی باید <= 1 باشد
        constraint = population[:, 0] + population[:, 1] - 1.0
        # تعیین اهداف: [هزینه کل، عرض مرز]
        out["F"] = np.column_stack([total_costs, total_boundary_width])
        out["G"] = constraint.reshape(-1, 1)

def optimize_threshold_scales(predicted_probs, false_pos_cost, false_neg_cost, population_size=20, num_generations=10):
    # ایجاد نمونه مسئله بهینه‌سازی با داده‌های ورودی
    problem_instance = ThresholdOptimizationProblem(predicted_probs, false_pos_cost, false_neg_cost)
    # تعریف الگوریتم NSGA-II با اندازه جمعیت مشخص شده
    nsga2_algo = NSGA2(pop_size=population_size)
    # اجرای بهینه‌سازی به تعداد نسل تعیین شده
    optimization_result = minimize(problem_instance, nsga2_algo, ('n_gen', num_generations), seed=1, verbose=False)
    # انتخاب بهترین راه‌حل بر اساس مرتب‌سازی lexicographic (ابتدا هزینه کل سپس عرض مرز)
    objectives = optimization_result.F
    best_index = np.lexsort((objectives[:, 1], objectives[:, 0]))[0]
    best_scale_fn, best_scale_fp = optimization_result.X[best_index]
    return best_scale_fn, best_scale_fp

if __name__ == "__main__":
    # داده‌های نمونه برای آزمایش برنامه
    sample_predicted_probs = np.array([0.1, 0.5, 0.8])
    sample_false_pos_cost = np.array([10, 20, 30])
    sample_false_neg_cost = np.array([5, 15, 25])
    best_scale_fn, best_scale_fp = optimize_threshold_scales(sample_predicted_probs,
                                                             sample_false_pos_cost,
                                                             sample_false_neg_cost,
                                                             population_size=20,
                                                             num_generations=10)
    print("Best scale for false negative:", best_scale_fn)
    print("Best scale for false positive:", best_scale_fp)
