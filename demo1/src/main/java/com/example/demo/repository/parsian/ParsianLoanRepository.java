package com.example.demo.repository.parsian;

import com.example.demo.entity.ParsianLoan;
import com.example.demo.repository.lendingclub.LendingClubCustomRepository;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface ParsianLoanRepository extends JpaRepository<ParsianLoan, Long> , ParsianLoanCustomRepository {

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
