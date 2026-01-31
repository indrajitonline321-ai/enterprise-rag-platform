package com.enterprise.rag.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.enterprise.rag.model.contactUS;

@Repository
public interface contactUsRepository extends JpaRepository<contactUS, Long> {
}
