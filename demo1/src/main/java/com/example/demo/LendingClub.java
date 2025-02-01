package com.example.demo;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDate;

@Entity
@Table(name = "loan")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
public class LendingClub {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long lenId;

    private Long id;
    private Integer memberId;
    private BigDecimal loanAmnt;
    private BigDecimal fundedAmnt;
    private BigDecimal fundedAmntInv;
    private String term;
    private BigDecimal intRate;
    private BigDecimal installment;
    private String grade;
    private String subGrade;
    private String empTitle;
    private String empLength;
    private String homeOwnership;
    private BigDecimal annualInc;
    private String verificationStatus;
    private LocalDate issueD;
    private String loanStatus;
    private Character pymntPlan;

    @Column(columnDefinition = "TEXT")
    private String url;

    @Column(name = "descc", columnDefinition = "TEXT")
    private String desc;

    private String purpose;
    private String title;
    private String zipCode;
    private String addrState;
    private BigDecimal dti;
    private Integer delinq_2yrs;
    private LocalDate earliestCrLine;
    private Integer fico_range_low;
    private Integer fico_range_high;
    private Integer inq_last_6mths;
    private Integer mthsSinceLastDelinq;
    private Integer mths_since_last_record;
    private Integer open_acc;
    private Integer pub_rec;
    private BigDecimal revol_bal;
    private BigDecimal revol_util;
    private Integer total_acc;
    private Character initial_list_status;
    private BigDecimal out_prncp;
    private BigDecimal out_prncp_inv;
    private BigDecimal total_pymnt;
    private BigDecimal total_pymnt_inv;
    private BigDecimal total_rec_prncp;
    private BigDecimal total_rec_int;
    private BigDecimal total_rec_late_fee;
    private BigDecimal recoveries;
    private BigDecimal collection_recovery_fee;
    private LocalDate last_pymnt_d;
    private BigDecimal last_pymnt_amnt;
    private LocalDate next_pymnt_d;
    private LocalDate last_credit_pull_d;
    private Integer last_fico_range_high;
    private Integer last_fico_range_low;
    private Integer collections_12_mths_ex_med;
    private Integer mths_Since_Last_Major_Derog;
    private Integer policy_code;
    private String application_type;
    private BigDecimal annual_inc_joint;
    private BigDecimal dti_joint;
    private String verification_status_joint;
    private Integer acc_now_delinq;
    private BigDecimal tot_coll_amt;
    private BigDecimal tot_cur_bal;
    private Integer open_acc_6m;
    private Integer open_act_il;
    private Integer open_il_12m;
    private Integer open_il_24m;
    private Integer mths_since_rcnt_il;
    private BigDecimal total_bal_il;
    private BigDecimal il_util;
    private Integer open_rv_12m;
    private Integer open_rv_24m;
    private BigDecimal max_bal_bc;
    private BigDecimal all_util;
    private BigDecimal total_rev_hi_lim;
    private Integer inq_fi;
    private Integer total_cu_tl;
    private Integer inq_last_12m;
    private Integer acc_open_past_24mths;
    private BigDecimal avg_cur_bal;
    private BigDecimal bc_open_to_buy;
    private BigDecimal bc_util;
    private Integer chargeoff_within_12_mths;
    private BigDecimal delinq_amnt;
    private Integer mo_sin_old_il_acct;
    private Integer mo_sin_old_rev_tl_op;
    private Integer mo_sin_rcnt_rev_tl_op;
    private Integer mo_sin_rcnt_tl;
    private Integer mort_acc;
    private Integer mths_since_recent_bc;
    private Integer mths_since_recent_bc_dlq;
    private Integer mths_since_recent_inq;
    private Integer mths_since_recent_revol_delinq;
    private Integer num_accts_ever_120_pd;
    private Integer num_actv_bc_tl;
    private Integer num_actv_rev_tl;
    private Integer num_bc_sats;
    private Integer num_bc_tl;
    private Integer num_il_tl;
    private Integer num_op_rev_tl;
    private Integer num_rev_accts;
    private Integer num_rev_tl_bal_gt_0;
    private Integer num_sats;
    private Integer num_tl_120dpd_2m;
    private Integer num_tl_30dpd;
    private Integer num_tl_90g_dpd_24m;
    private Integer num_tl_op_past_12m;
    private BigDecimal pct_tl_nvr_dlq;
    private BigDecimal percent_bc_gt_75;
    private Integer pub_rec_bank_ruptcies;
    private Integer tax_liens;
    private BigDecimal tot_hi_cred_lim;
    private BigDecimal total_bal_ex_mort;
    private BigDecimal total_bc_limit;
    private BigDecimal total_Il_High_Credit_Limit;
    private BigDecimal revol_bal_joint;
    private Integer sec_app_fico_range_low;
    private Integer sec_app_fico_range_high;
    private LocalDate sec_app_earliest_cr_line;
    private Integer sec_app_inq_last_6mths;
    private Integer sec_app_mort_acc;
    private Integer sec_app_open_acc;
    private BigDecimal sec_app_revol_util;
    private Integer sec_app_open_act_il;
    private Integer sec_app_num_rev_accts;
    private Integer sec_app_chargeoff_within_12_mths;
    private Integer sec_app_collections_12_mths_ex_med;
    private Integer sec_app_mths_since_last_major_derog;
    private Character hardship_flag;
    private String hardship_type;
    private String hardship_reason;
    private String hardship_status;
    private Integer deferral_term;
    private BigDecimal hardship_amount;
    private LocalDate hardship_start_date;
    private LocalDate hardship_end_date;
    private LocalDate payment_plan_start_date;
    private Integer hardship_length;
    private Integer hardship_dpd;
    private String hardship_loan_status;
    private BigDecimal orig_projected_additional_accrued_interest;
    private BigDecimal hardship_payoff_balance_amount;
    private BigDecimal hardship_last_payment_amount;
    private String disbursement_method;
    private Character debt_settlement_flag;
    private LocalDate debt_settlement_flag_date;
    private String settlement_status;
    private LocalDate settlement_date;
    private BigDecimal settlement_amount;
    private BigDecimal settlement_percentage;
    private Integer settlement_term;
}
