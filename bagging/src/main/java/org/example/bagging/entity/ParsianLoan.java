package org.example.bagging.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "PARSIAN_LOAN")
public class ParsianLoan {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "ID")
    private Long id;

    @Column(name = "INSERT_SYSDATE", nullable = false)
    private LocalDateTime insertSysdate;

    @Column(name = "branch_code", nullable = false)
    private Integer branchCode;

    @Column(name = "branchname", columnDefinition = "VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    private String branchName;

    @Column(name = "loan_file_numberr")
    private Long clientId;

    @Column(name = "CLIENT_ID")
    private Integer clientIdNumber;

    @Column(name = "sit_flag")
    private Character sitFlag;

    @Column(name = "interest_rate", precision = 19, scale = 2)
    private BigDecimal interestRate;

    @Column(name = "total_repayment_up_to_now", precision = 28, scale = 8)
    private BigDecimal totalRepaymentUpToNow;

    @Column(name = "COMMISSION_AMOUNT_REMAIN", precision = 28, scale = 8)
    private BigDecimal commissionAmountRemain;

    @Column(name = "CHARGE", precision = 28, scale = 8)
    private BigDecimal charge;

    @Column(name = "DISCOUNT", precision = 28, scale = 8)
    private BigDecimal discount;

    @Column(name = "ADVANCE_PAY_TO_TOTAL_CASH", precision = 28, scale = 8)
    private BigDecimal advancePayToTotalCash;

    @Column(name = "ADVANCE_PAY_TO_REMAIN_NON_CASH", precision = 28, scale = 8)
    private BigDecimal advancePayToRemainNonCash;

    @Column(name = "is_installment")
    private Character isInstallment;

    @Column(name = "interest_sum", precision = 28, scale = 8)
    private BigDecimal interestSum;

    @Column(name = "INSTALLMENT_NUMBER_REMAIN")
    private Integer installmentNumberRemain;

    @Column(name = "RECEIVABLE_INSTALLMENT_NUMBER")
    private Integer receivableInstallmentNumber;

    @Column(name = "FIRST_PASSED")
    private LocalDate firstPassed;

    @Column(name = "TOTAL_PAYMENT_UP_TO_NOW", precision = 28, scale = 8)
    private BigDecimal totalPaymentUpToNow;

    @Column(name = "FINALIZED_LOAN_AMOUNT", precision = 28, scale = 8)
    private BigDecimal finalizedLoanAmount;

    @Column(name = "PENALTY", precision = 28, scale = 8)
    private BigDecimal penalty;

    @Column(name = "FIRST_PAYMENT_DATE_IN_DU")
    private LocalDate firstPaymentDateInDu;

    @Column(name = "PRINCIPAL_SUM", precision = 28, scale = 8)
    private BigDecimal principalSum;

    @Column(name = "ADVANCE_PAY", precision = 28, scale = 8)
    private BigDecimal advancePay;

    @Column(name = "SIT_DURATION")
    private Integer sitDuration;

    @Column(name = "sit_duration_day")
    private Integer sitDurationDay;

    @Column(name = "sit_distribute_phases")
    private Integer sitDistributePhases;

    @Column(name = "sit_fast_receive_percent")
    private Float sitFastReceivePercent;

    @Column(name = "frequency")
    private Integer frequency;

    @Column(name = "CUSTOMER_OBLIGATION_AMOUNT", precision = 28, scale = 8)
    private BigDecimal customerObligationAmount;

    @Column(name = "CUSTOMER_SHARE_CASH_AMOUNT", precision = 28, scale = 8)
    private BigDecimal customerShareCashAmount;

    @Column(name = "CUSTOMER_SHARE_NON_CASH_AMOUNT", precision = 28, scale = 8)
    private BigDecimal customerShareNonCashAmount;

    @Column(name = "BANK_SHARE_CASH_AMOUNT", precision = 28, scale = 8)
    private BigDecimal bankShareCashAmount;

    @Column(name = "BANK_SHARE_NON_CASH_AMOUNT", precision = 28, scale = 8)
    private BigDecimal bankShareNonCashAmount;

    @Column(name = "First_OverDue")
    private LocalDate firstOverDue;

    @Column(name = "LOAN_DURATION_DAY")
    private Integer loanDurationDay;

    @Column(name = "Loan_File_Number")
    private Long loanFileNumber;

    @Column(name = "CreateDate")
    private LocalDate createDate;

    @Column(name = "LONG_TITLE", columnDefinition = "VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    private String longTitle;

    @Column(name = "Status", columnDefinition = "VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    private String status;

    @Column(name = "contract", columnDefinition = "VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    private String contract;

    @Column(name = "APPROVAL_AMOUNT", precision = 28, scale = 8)
    private BigDecimal approvalAmount;

    @Column(name = "TITLE", columnDefinition = "VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    private String title;

    @Column(name = "inc_COMMISSION_AMOUNT", precision = 28, scale = 8)
    private BigDecimal incCommissionAmount;

    @Column(name = "INTEREST_AMOUNT", precision = 28, scale = 8)
    private BigDecimal interestAmount;

    @Column(name = "OBLIGATION_PENALTY", precision = 28, scale = 8)
    private BigDecimal obligationPenalty;

    @Column(name = "PASSED_DATE")
    private LocalDate passedDate;

    @Column(name = "PENALTY_INTEREST", precision = 28, scale = 8)
    private BigDecimal penaltyInterest;

    @Column(name = "TO_DUE_DATE", precision = 28, scale = 8)
    private BigDecimal toDueDate;

    @Column(name = "TO_END_OF_MONTH", precision = 28, scale = 8)
    private BigDecimal toEndOfMonth;

    @Column(name = "DUE_DATE")
    private LocalDate dueDate;
}
