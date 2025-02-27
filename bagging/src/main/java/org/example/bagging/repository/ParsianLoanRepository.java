package org.example.bagging.repository;


import org.example.bagging.entity.ParsianLoan;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ParsianLoanRepository extends JpaRepository<ParsianLoan, Long>, ParsianLoanCustomRepository {

    // جستجو بر اساس branchCode
    List<ParsianLoan> findByBranchCode(Integer branchCode);

    // جستجو بر اساس ClientId
    List<ParsianLoan> findByClientId(Long clientId);

    // جستجو بر اساس وضعیت (Status)
    List<ParsianLoan> findByStatus(String status);

    // جستجو بر اساس محدوده نرخ سود
    List<ParsianLoan> findByInterestRateBetween(Double minRate, Double maxRate);


    List<String> findDistinctStatusBy();



}
