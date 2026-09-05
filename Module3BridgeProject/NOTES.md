# The Vendor Onboarding Desk

## a. Where did the budget force a real trade-off? Which supplier did you decide not to fully check, and why that one?
With only 9 lookups for 7 suppliers, I couldn't afford to waste a single query. I chose to skip deep secondary lookups on *C & V Works ApS* because it was a low-value European vendor with a clean initial profile. I preferred to save those extra lookups for messy multi-hit suppliers (like Siemens variants) where mixing up the legal entity could misdirect a multi-million-euro payment. I chose spending my lookup budget on high-ambiguity names over over-checking a straightforward, low-risk supplier.

---

## b. What sanctions threshold did you set, and why that number? Show the score that made you pick it. Then answer honestly: what would it have cost you to be wrong in each direction?
I set the threshold at **0.85**. Anything lower (like 0.71 to 0.81) kept catching false positives because generic words like "Trading" or "Company" overlap with everything in the database. 
- **Cost of a False Negative (Missing a real sanction):** Criminal charges, massive legal fines, and severe penalties for financing a blocked entity.
- **Cost of a False Positive (Blocking an honest supplier):** Delayed payments and angry suppliers, which is annoying for business and hurts relations, but it doesn't land anyone in prison. Because the risk is so heavily skewed, I drew the line high enough to stop real threats without accidentally freezing innocent vendors.

---

## c. Where did your agent nearly get it wrong? Every one of you will hit at least one of the four traps. Which one, what did it output first, what did you change?
I hit **Trap #1 ("The register found it, so that's our supplier")**. 
When the search returned 5 different legal entities under the same name, the agent initially just grabbed the first result (`data[0]`) and confidently output an `APPROVE` verdict for a company nobody actually asked for. 
To fix it, I updated the logic so the agent is forced to check if there is *strictly one* matching candidate. If the register returns multiple options, it’s not allowed to guess—it has to switch the verdict to `CONDITIONS` and ask Procurement for the exact LEI or registration number.