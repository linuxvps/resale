package com.example.demo;

import com.example.demo.bagging.service.BaggingService;
import com.example.demo.hambastegi.DataDistributionAnalyzer;
import com.example.demo.hambastegi.OutlierAnalyzer;
import com.example.demo.hambastegi.PearsonCorrelationService;
import com.example.demo.savecsv.ParsianLoanService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
//--add-opens java.base/java.lang=ALL-UNNAMED
@SpringBootApplication
public class Demo1Application implements CommandLineRunner {

    @Autowired
    private CreditRiskService creditRiskService;


    @Autowired
    private PearsonCorrelationService pearsonCorrelationService;

    @Autowired
    private DataDistributionAnalyzer dataDistributionAnalyzer;

    @Autowired
    private OutlierAnalyzer dataOutlierAnalyzer;

    @Autowired
    private BaggingService baggingService;

    @Autowired
    private ParsianLoanService parsianLoanService;


    public static void main(String[] args) {
        SpringApplication.run(Demo1Application.class, args);
    }

    @Override
    public void run(String... args) throws Exception {

//        creditRiskService.calcAmarTosifiVaMoteghayerTahghigAndCreateReport();

//        pearsonCorrelationService.calcAndCreateReportOfPearsonsCorrelation();


//        dataOutlierAnalyzer.calcOutlierAnalyzer();
        baggingService.calcBagging();

//        parsianLoanService.processCsvFiles("C:\\Users\\nima\\Desktop\\lendingClub\\parianLoan");


    }
}
