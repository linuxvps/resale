package com.example.demo;

import com.example.demo.hambastegi.DataDistributionAnalyzer;
import com.example.demo.hambastegi.PearsonCorrelationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

@SpringBootApplication
public class Demo1Application implements CommandLineRunner {

    @Autowired
    private CreditRiskService creditRiskService;


    @Autowired
    private PearsonCorrelationService pearsonCorrelationService;

    @Autowired
    private DataDistributionAnalyzer dataDistributionAnalyzer;


    public static void main(String[] args) {
        SpringApplication.run(Demo1Application.class, args);
    }

    @Override
    public void run(String... args) throws Exception {

//        creditRiskService.calcAmarTosifiVaMoteghayerTahghigAndCreateReport();

//        pearsonCorrelationService.calcAndCreateReportOfPearsonsCorrelation();

        Map<String, Object> analysisResult = dataDistributionAnalyzer.analyzeColumnDistribution("annual_inc");
        analysisResult.forEach((key, value) -> System.out.println(key + ": " + value));


    }
}
