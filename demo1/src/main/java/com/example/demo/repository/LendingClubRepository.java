package com.example.demo.repository;

import com.example.demo.LendingClub;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface LendingClubRepository extends JpaRepository<LendingClub, Long>, LendingClubCustomRepository {
}

