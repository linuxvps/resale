package org.example.bagging;

import org.example.bagging.service.BaggingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BaggingApplication implements CommandLineRunner {

    public static void main(String[] args) {
        SpringApplication.run(BaggingApplication.class, args);
    }

    @Autowired
    private BaggingService baggingService;


    @Override
    public void run(String... args) throws Exception {
        baggingService.calcBagging();
    }
}
