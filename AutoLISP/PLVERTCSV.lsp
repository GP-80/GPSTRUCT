(defun C:PLVERTCSV (/ ent pldata ptlist csvfile filepath closed)
  (setq ent (car (entsel "\nSelect polyline: ")))
  
  (if (= (cdr (assoc 0 (entget ent))) "LWPOLYLINE")
    (progn
      ; Get polyline data
      (setq pldata (entget ent))
      
      ; Check if polyline is closed
      (setq closed (= (cdr (assoc 70 pldata)) 1))
      
      ; Initialize point list
      (setq ptlist nil)
      
      ; Extract vertices
      (foreach vertex pldata
        (if (= (car vertex) 10)  ; 10 is the group code for vertex coordinates
          (setq ptlist (cons (cdr vertex) ptlist))
        )
      )
      
      ; Reverse to get correct order and remove last vertex if closed
      (if closed
          (setq ptlist (cdr ptlist))
          (setq ptlist ptlist)
      )
      
      ; Get file path from user using file dialog
      (setq filepath (getfiled "Save CSV File" "" "csv" 1))
      
      ; Check if user selected a file
      (if filepath
        (progn
          ; Open CSV file for writing
          (setq csvfile (open filepath "w"))
          
          ; Write header
          (write-line "X,Y" csvfile)
          
          ; Write coordinates
          (foreach pt ptlist
            (write-line (strcat (rtos (car pt) 2 8) "," (rtos (cadr pt) 2 8)) csvfile)
          )
          
          ; Close file
          (close csvfile)
          
          (princ (strcat "\nCoordinates exported to: " filepath))
        )
        (princ "\nFile export cancelled.")
      )
    )
    (princ "\nSelected entity is not a lightweight polyline.")
  )
  (princ)
)
