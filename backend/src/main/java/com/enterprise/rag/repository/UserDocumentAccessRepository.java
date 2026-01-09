package com.enterprise.rag.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.enterprise.rag.model.UserDocumentAccess;

@Repository
public interface UserDocumentAccessRepository extends JpaRepository<UserDocumentAccess, Long> {
    List<UserDocumentAccess> findByUserId(long userId);
}

