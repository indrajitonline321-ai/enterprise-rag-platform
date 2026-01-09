package com.enterprise.rag.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.enterprise.rag.model.Document;


@Repository
public interface DocumentRepository extends JpaRepository<Document, Long> {
    
    @Query("SELECT DISTINCT d FROM Document d " +
           "JOIN UserDocumentAccess uda ON d.id = uda.document.id " +
           "WHERE uda.user.id = :userId")
    List<Document> findByUserId(@Param("userId") long userId);
}