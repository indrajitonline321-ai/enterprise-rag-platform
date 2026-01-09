package com.enterprise.rag.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.enterprise.rag.model.User;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User findByName(String name);
}

