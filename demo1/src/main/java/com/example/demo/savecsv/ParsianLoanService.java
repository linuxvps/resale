package com.example.demo.savecsv;

import com.example.demo.entity.ParsianLoan;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Stream;

@Service
public class ParsianLoanService {

    @PersistenceContext
    private EntityManager entityManager;

    @Autowired
    private com.example.demo.savecsv.savetrx.entityManagerTrx entityManagerTrx;

    private static final int EXPECTED_COLUMNS = 51;  // تعداد ستون‌های جدول
    private static final int BATCH_SIZE = 1000;  // هر 1000 رکورد یکبار ذخیره شود

    public void processCsvFiles(String folderPath) {
        try (Stream<Path> paths = Files.list(Paths.get(folderPath))) {
            paths.parallel()
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".csv"))
                    .forEach(this::processCsvFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void processCsvFile(Path filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()));
             CSVReader csvReader = new CSVReader(br)) {

            String[] header = csvReader.readNext(); // خواندن هدر
            List<ParsianLoan> batch = new ArrayList<>();

            String[] line;
            while ((line = csvReader.readNext()) != null) {
                ParsianLoan entity = mapToEntity(normalizeColumns(line, EXPECTED_COLUMNS));
                batch.add(entity);

                if (batch.size() >= BATCH_SIZE) {
                    saveBatch(batch);
                    batch.clear();
                }
            }

            if (!batch.isEmpty()) {
                saveBatch(batch);
            }

            System.out.println("✅ Processed file: " + filePath.getFileName());

        } catch (IOException | CsvValidationException e) {
            e.printStackTrace();
        }
    }

    public void saveBatch(List<ParsianLoan> batch) {
        entityManagerTrx.saveBatchParsian(batch);
    }

    private ParsianLoan mapToEntity(String[] columns) {
        ParsianLoan entity = new ParsianLoan();
        Field[] fields = Arrays.stream(ParsianLoan.class.getDeclaredFields())
                .filter(field -> !field.getName().equals("id")) // حذف id که Auto Increment است
                .toArray(Field[]::new);

        for (int i = 0; i < fields.length && i < columns.length; i++) {
            fields[i].setAccessible(true);
            try {
                Object newValue = switch (fields[i].getType().getSimpleName()) {
                    case "Integer" -> parseInt(columns[i]);
                    case "Long" -> parseLong(columns[i]);
                    case "BigDecimal" -> parseBigDecimal(columns[i]);
                    case "LocalDate" -> parseLocalDate(columns[i]);
                    case "LocalDateTime" -> parseLocalDateTime(columns[i]);
                    case "Character" -> parseCharacter(columns[i]);
                    case "Float" -> parseFloat(columns[i]);
                    default -> cleanString(columns[i]);
                };
                fields[i].set(entity, newValue);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
        return entity;
    }
    private Float parseFloat(String value) {
        if (value == null || value.trim().isEmpty()) return null;
        try {
            return Float.parseFloat(value.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }


    private static Character parseCharacter(String value) {
        return (value == null || value.trim().isEmpty()) ? null : value.trim().charAt(0);
    }

    private Integer parseInt(String value) {
        if (value == null || value.isEmpty()) return null;
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private Long parseLong(String value) {
        if (value == null || value.isEmpty()) return null;
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private BigDecimal parseBigDecimal(String value) {
        if (value == null || value.isEmpty()) return null;
        try {
            return new BigDecimal(value.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private LocalDate parseLocalDate(String value) {
        if (value == null || value.trim().isEmpty()) return null;
        try {
            return LocalDate.parse(value.trim(), DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        } catch (Exception e) {
            return null;
        }
    }

    private LocalDateTime parseLocalDateTime(String value) {
        if (value == null || value.trim().isEmpty()) return null;
        try {
            return LocalDateTime.parse(value.trim(), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        } catch (Exception e) {
            return null;
        }
    }

    private String cleanString(String value) {
        return (value == null) ? null : value.replaceAll("\"", "").trim();
    }

    private String[] normalizeColumns(String[] row, int expectedLength) {
        String[] columns = new String[expectedLength];
        Arrays.fill(columns, null);

        for (int i = 0; i < Math.min(row.length, expectedLength); i++) {
            columns[i] = row[i].trim().isEmpty() ? null : row[i].trim();
        }

        return columns;
    }
}
