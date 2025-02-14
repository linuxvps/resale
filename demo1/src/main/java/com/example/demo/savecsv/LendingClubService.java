package com.example.demo.savecsv;

import com.example.demo.entity.LendingClub;
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
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;
import java.util.stream.Stream;

@Service
public class LendingClubService {

    @PersistenceContext
    private EntityManager entityManager;

    @Autowired
    private com.example.demo.savecsv.savetrx.entityManagerTrx entityManagerTrx;

    private static final int EXPECTED_COLUMNS = 151;
    private static final int BATCH_SIZE = 5000;  // هر 5000 رکورد یکبار ذخیره شود

    public void processCsvFiles(String folderPath) {
        try (Stream<Path> paths = Files.list(Paths.get(folderPath))) {
            paths.parallel() // 🚀 پردازش همزمان چندین فایل
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
            List<LendingClub> batch = new ArrayList<>();

            String[] line;
            while ((line = csvReader.readNext()) != null) {
                LendingClub entity = mapToEntity(normalizeColumns(line, EXPECTED_COLUMNS));
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

    public void saveBatch(List<LendingClub> batch) {
        entityManagerTrx.saveBatch(batch);
    }

    private LendingClub mapToEntity(String[] columns) {
        LendingClub entity = new LendingClub();
        Field[] fields = Arrays.stream(LendingClub.class.getDeclaredFields())
                .filter(field -> !field.getName().equals("lenId")) // حذف lenId
                .toArray(Field[]::new);

        for (int i = 0; i < fields.length && i < columns.length; i++) {
            fields[i].setAccessible(true);
            try {
                Object newValue = switch (fields[i].getType().getSimpleName()) {
                    case "Integer" -> parseInt(columns[i]);
                    case "Long" -> parseLong(columns[i]);
                    case "BigDecimal" -> parseBigDecimal(columns[i]);
                    case "LocalDate" -> parseMonthYearToDate(columns[i]);
                    case "Character" -> parseCharacter(columns[i]);
                    default -> cleanString(columns[i]);
                };
                fields[i].set(entity, newValue);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
        return entity;
    }

    private static Character parseCharacter(String value) {
        return (value == null || value.trim().isEmpty()) ? null : value.trim().charAt(0);
    }

    private Integer parseInt(String value) {
        if (value == null || value.isEmpty()) return null;
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            try {
                return (int) Double.parseDouble(value.trim());
            } catch (NumberFormatException ex) {
                return null;
            }
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

    private LocalDate parseMonthYearToDate(String dateStr) {
        if (dateStr == null || dateStr.trim().isEmpty()) return null;
        try {
            return YearMonth.parse(dateStr.trim(), DateTimeFormatter.ofPattern("MMM-yyyy", Locale.ENGLISH))
                    .atDay(1);
        } catch (DateTimeParseException e) {
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
