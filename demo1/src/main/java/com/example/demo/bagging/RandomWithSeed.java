package com.example.demo.bagging;

import java.util.Random;

public class RandomWithSeed {
    public static void main(String[] args) {
        Random rand = new Random(42); // مقدار ثابت برای Seed
        System.out.println(rand.nextInt(100)); // عدد تصادفی بین 0 تا 99
    }
}