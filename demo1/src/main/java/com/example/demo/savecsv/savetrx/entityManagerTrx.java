package com.example.demo.savecsv.savetrx;

import com.example.demo.entity.LendingClub;
import com.example.demo.entity.ParsianLoan;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class entityManagerTrx {


    @PersistenceContext
    private EntityManager entityManager;

    @Transactional
    public void saveBatch(List<LendingClub> batch) {
        for (int i = 0; i < batch.size(); i++) {
            entityManager.persist(batch.get(i));

            if (i % 100 == 0) {  // 🚀 هر 100 رکورد `flush` و `clear` کنیم تا حافظه بیش از حد مصرف نشود
                entityManager.flush();
                entityManager.clear();
            }
        }
        entityManager.flush();
        entityManager.clear();
    }

    @Transactional
    public void saveBatchParsian(List<ParsianLoan> batch) {
        for (int i = 0; i < batch.size(); i++) {
            entityManager.persist(batch.get(i));

            if (i % 100 == 0) {  // 🚀 هر 100 رکورد `flush` و `clear` کنیم تا حافظه بیش از حد مصرف نشود
                entityManager.flush();
                entityManager.clear();
            }
        }
        entityManager.flush();
        entityManager.clear();
    }
}
