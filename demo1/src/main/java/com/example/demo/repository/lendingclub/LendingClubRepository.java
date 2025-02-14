package com.example.demo.repository.lendingclub;

import com.example.demo.entity.LendingClub;
import jakarta.transaction.Transactional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface LendingClubRepository extends JpaRepository<LendingClub, Long>, LendingClubCustomRepository {

    @Modifying
    @Transactional
    @Query("UPDATE LendingClub l SET l.isDeleted = true WHERE l.lenId IN (:ids)")
    void updateIsDeletedForIds(List<Long> ids);
}

