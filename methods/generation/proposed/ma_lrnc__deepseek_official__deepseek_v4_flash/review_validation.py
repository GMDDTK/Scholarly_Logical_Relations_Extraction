from .contracts.paper_profile import PaperProfileContent
from .contracts.profile_review import (
    ProfileReviewNeedsRevision,
    ProfileReviewResult,
)


class ProfileReviewLocationError(ValueError):
    pass


def validate_profile_review_locations(
    review: ProfileReviewResult,
    current_profile: PaperProfileContent,
) -> None:
    if not isinstance(review, ProfileReviewNeedsRevision):
        return
    for issue_index, issue in enumerate(review.issues):
        for location_index, location in enumerate(issue.target_locations):
            if location.item_index is None:
                continue
            items = getattr(current_profile, location.field.value)
            if location.item_index >= len(items):
                raise ProfileReviewLocationError(
                    "issues"
                    f"[{issue_index}].target_locations[{location_index}] "
                    "does not reference an existing profile item"
                )
