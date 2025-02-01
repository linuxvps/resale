package com.example.demo;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
interface LendingClubRepository extends JpaRepository<LendingClub, Long> ,LendingClubCustomRepository{
}

