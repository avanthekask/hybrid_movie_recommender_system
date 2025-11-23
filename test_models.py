try:
    import models.phase2_contentmodel as c
    print("phase2 imported:", c)
except Exception as e:
    print("phase2 FAILED:", e)

try:
    import models.phase3_collabfiltering as c2
    print("phase3 imported:", c2)
except Exception as e:
    print("phase3 FAILED:", e)

try:
    import models.phase4_hybridfusion as c3
    print("phase4 imported:", c3)
except Exception as e:
    print("phase4 FAILED:", e)
